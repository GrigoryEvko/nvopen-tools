// Function: sub_248AF20
// Address: 0x248af20
//
void __fastcall sub_248AF20(
        __int64 a1,
        __int64 a2,
        unsigned __int8 (__fastcall *a3)(_QWORD *, __int64, __int64),
        _QWORD *a4,
        void (__fastcall *a5)(__int64, unsigned __int64, _QWORD, __int64, _QWORD *),
        __int64 a6)
{
  int v6; // eax
  int v7; // r12d
  unsigned __int64 v8; // r12
  char *v10; // rax
  char *v11; // rbx
  _DWORD *v12; // rcx
  unsigned __int64 *v13; // r12
  char *v14; // rbx
  char *v15; // r13
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // rdi
  _QWORD *v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  char *v21; // rax
  signed __int64 v22; // r13
  __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rdx
  int i; // r15d
  int v28; // eax
  __int64 v29; // r13
  int v30; // r12d
  __int64 v31; // rbx
  __int64 v32; // rcx
  _QWORD *v33; // r8
  unsigned __int64 *v34; // rdi
  __int64 v35; // r9
  int v36; // r12d
  unsigned __int64 *v37; // rsi
  unsigned __int64 v38; // rdi
  int v39; // eax
  int v40; // edx
  int v41; // ebx
  int v42; // eax
  __int64 v43; // rcx
  unsigned __int64 *v44; // rdx
  int v45; // r13d
  __int64 v46; // r12
  int v47; // r14d
  _QWORD *v48; // rdi
  unsigned __int64 v49; // rsi
  unsigned __int64 *v50; // rbx
  unsigned __int64 *v51; // r12
  unsigned __int64 *v52; // r13
  int v54; // [rsp+14h] [rbp-DCh]
  int v55; // [rsp+18h] [rbp-D8h]
  int v56; // [rsp+20h] [rbp-D0h]
  int v57; // [rsp+24h] [rbp-CCh]
  int v59; // [rsp+30h] [rbp-C0h]
  int v60; // [rsp+38h] [rbp-B8h]
  unsigned __int64 *v62; // [rsp+40h] [rbp-B0h]
  int v64; // [rsp+48h] [rbp-A8h]
  __int64 v66; // [rsp+50h] [rbp-A0h]
  int v67; // [rsp+58h] [rbp-98h]
  int v68; // [rsp+5Ch] [rbp-94h]
  _QWORD v69[2]; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v70[2]; // [rsp+70h] [rbp-80h] BYREF
  void *src; // [rsp+80h] [rbp-70h] BYREF
  char *v72; // [rsp+88h] [rbp-68h]
  char *v73; // [rsp+90h] [rbp-60h]
  unsigned __int64 *v74; // [rsp+A0h] [rbp-50h] BYREF
  unsigned __int64 *v75; // [rsp+A8h] [rbp-48h]
  unsigned __int64 *v76; // [rsp+B0h] [rbp-40h]

  v68 = *(_DWORD *)(a1 + 8);
  v6 = *(_DWORD *)(a2 + 8) + v68;
  v56 = *(_DWORD *)(a2 + 8);
  v55 = v6;
  if ( !v6 )
    return;
  v7 = 2 * v6 + 1;
  if ( (unsigned __int64)v7 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = 4LL * v7;
  v72 = 0;
  v10 = (char *)sub_22077B0(v8);
  v11 = &v10[v8];
  src = v10;
  v12 = v10;
  v73 = &v10[v8];
  if ( &v10[v8] != v10 )
    v12 = memset(v10, 255, v8);
  v72 = v11;
  v54 = v55 + 1;
  v12[v55 + 1] = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  if ( v55 >= 0 )
  {
    v57 = 0;
    v13 = 0;
LABEL_73:
    v17 = (unsigned __int64)&v74;
    sub_248ACA0((unsigned __int64)&v74, v13, (__int64)&src);
    v14 = (char *)src;
LABEL_16:
    v23 = (unsigned int)(v54 - v57);
    v59 = v54 - v57;
    v60 = -v57;
LABEL_30:
    v25 = v59;
LABEL_19:
    v26 = (__int64)src;
    for ( i = *((_DWORD *)src + v25); ; i = v24 + 1 )
    {
      v28 = i - v60;
      if ( v68 > i )
      {
        if ( v56 <= v28 )
        {
LABEL_27:
          *(_DWORD *)&v14[4 * v59 - 4] = i;
          goto LABEL_28;
        }
        v29 = 16LL * v28;
        v30 = i;
        v31 = 16 * (i - (__int64)v28);
        while ( 1 )
        {
          v17 = (unsigned __int64)a4;
          v23 = v31 + *(_QWORD *)a1 + v29 + 8;
          if ( !a3(a4, v23, v29 + *(_QWORD *)a2 + 8) )
          {
LABEL_26:
            v14 = (char *)src;
            i = v30;
            goto LABEL_27;
          }
          v28 = ++v30 - v60;
          if ( v68 == v30 )
            break;
          v29 += 16;
          if ( v30 == v56 + v60 )
            goto LABEL_26;
        }
        v14 = (char *)src;
        i = v30;
      }
      v32 = (unsigned int)v59;
      v26 = v59 - 1;
      *(_DWORD *)&v14[4 * v26] = i;
      if ( v56 <= v28 )
      {
        v70[1] = 0;
        v70[0] = (unsigned __int64)&src;
        if ( *(_DWORD *)(a2 + 8) )
          sub_2484F60((__int64)v70, a2, v26, (unsigned int)v59, v19, v20);
        v33 = v69;
        v69[1] = 0;
        v69[0] = v70;
        v34 = v70;
        v35 = *(unsigned int *)(a1 + 8);
        if ( (_DWORD)v35 )
        {
          sub_2484F60((__int64)v69, a1, v26, v32, (__int64)v69, v35);
          v34 = (unsigned __int64 *)v69[0];
        }
        v64 = -1431655765 * (v75 - v74) - 1;
        if ( v56 > 0 || v68 > 0 )
        {
          v36 = v56;
          v37 = v34;
          v62 = &v74[3 * v64];
          while ( 1 )
          {
            v38 = *v62;
            v39 = v68 - v36;
            if ( v68 - v36 + v64 )
            {
              v40 = v39 - 1;
              v41 = *(_DWORD *)(v38 + 4LL * (v55 + v39 - 1));
              if ( v64 != v39 )
              {
                v42 = v39 + 1;
                if ( *(_DWORD *)(v38 + 4LL * (v55 + v42)) > v41 )
                {
                  v41 = *(_DWORD *)(v38 + 4LL * (v55 + v42));
                  v40 = v42;
                }
              }
            }
            else
            {
              v40 = v39 + 1;
              v41 = *(_DWORD *)(v38 + 4LL * (v39 + 1 + v55));
            }
            v67 = v41 - v40;
            if ( v41 - v40 < v36 )
            {
              v43 = (unsigned int)v68;
              if ( v68 > v41 )
              {
                v44 = v37;
                v66 = 2 * (v68 - (__int64)v36);
                v45 = v36;
                v46 = 2 * (v36 - 1LL);
                v47 = v45;
                while ( 1 )
                {
                  --v47;
                  v48 = (_QWORD *)(v46 * 8 + v70[0]);
                  v49 = v44[v46 + v66];
                  v46 -= 2;
                  a5(a6, v49, *v48, v43, v33);
                  if ( v47 + v68 - v45 <= v41 || v67 >= v47 )
                    break;
                  v44 = (unsigned __int64 *)v69[0];
                }
                v37 = (unsigned __int64 *)v69[0];
              }
            }
            if ( !v64 )
              break;
            v62 -= 3;
            --v64;
            if ( v41 <= 0 && v67 <= 0 )
              break;
            v68 = v41;
            v36 = v67;
          }
          v34 = v37;
        }
        if ( v34 != v70 )
          _libc_free((unsigned __int64)v34);
        if ( (void **)v70[0] != &src )
          _libc_free(v70[0]);
        v50 = v75;
        v51 = v74;
        if ( v75 != v74 )
        {
          do
          {
            if ( *v51 )
              j_j___libc_free_0(*v51);
            v51 += 3;
          }
          while ( v50 != v51 );
          v51 = v74;
        }
        if ( v51 )
          j_j___libc_free_0((unsigned __int64)v51);
        break;
      }
LABEL_28:
      v60 += 2;
      v59 += 2;
      if ( v60 > v57 )
      {
        ++v57;
        v13 = v75;
        if ( v55 < v57 )
        {
          v52 = v74;
          if ( v74 != v75 )
          {
            do
            {
              if ( *v52 )
                j_j___libc_free_0(*v52);
              v52 += 3;
            }
            while ( v52 != v13 );
            v52 = v74;
          }
          if ( v52 )
            j_j___libc_free_0((unsigned __int64)v52);
          break;
        }
        if ( v75 == v76 )
          goto LABEL_73;
        v14 = (char *)src;
        if ( v75 )
        {
          v75[2] = 0;
          v15 = v72;
          *v13 = 0;
          v13[1] = 0;
          v16 = v15 - v14;
          if ( v16 )
          {
            if ( v16 > 0x7FFFFFFFFFFFFFFCLL )
              sub_4261EA(v17, v23, v26);
            v17 = v16;
            v18 = (_QWORD *)sub_22077B0(v16);
          }
          else
          {
            v18 = 0;
          }
          *v13 = (unsigned __int64)v18;
          v14 = (char *)src;
          v13[1] = (unsigned __int64)v18;
          v21 = v72;
          v13[2] = (unsigned __int64)v18 + v16;
          v22 = v21 - v14;
          if ( v21 != v14 )
          {
            v17 = (unsigned __int64)v18;
            v18 = memmove(v18, v14, v21 - v14);
          }
          v13[1] = (unsigned __int64)v18 + v22;
          v13 = v75;
        }
        v75 = v13 + 3;
        goto LABEL_16;
      }
      v14 = (char *)src;
      if ( -v57 == v60 )
        goto LABEL_30;
      v23 = (unsigned int)v59;
      v17 = (unsigned int)v60;
      v24 = *((_DWORD *)src + v59 - 2);
      if ( v57 != v60 )
      {
        v25 = v59;
        if ( *((_DWORD *)src + v59) > v24 )
          goto LABEL_19;
      }
    }
  }
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}
