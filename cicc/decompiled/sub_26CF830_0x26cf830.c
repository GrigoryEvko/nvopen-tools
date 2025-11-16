// Function: sub_26CF830
// Address: 0x26cf830
//
void __fastcall sub_26CF830(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v7; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  _QWORD *v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 *v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rbx
  size_t v20; // r14
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r8
  _QWORD *v23; // rax
  _QWORD *v24; // rsi
  __int64 v25; // rdi
  _QWORD *v26; // rbx
  size_t v27; // r14
  unsigned __int64 v28; // rdi
  _QWORD *v29; // r9
  __int64 v30; // r8
  _QWORD *v31; // rax
  _QWORD *v32; // rsi
  __int64 v33; // rdi
  unsigned int v34; // esi
  __int64 v35; // rdi
  unsigned int v36; // eax
  size_t *v37; // rcx
  size_t v38; // rdx
  int v39; // r10d
  size_t *v40; // r9
  int v41; // eax
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rbx
  _QWORD *v45; // r13
  unsigned __int64 v46; // r14
  __int64 v47; // r12
  __int64 v48; // rax
  const char *v49; // rax
  __int64 v50; // rdx
  int v51; // edx
  int v52; // edx
  __int64 v53; // rdi
  unsigned int v54; // eax
  size_t v55; // rcx
  int v56; // eax
  __int64 v57; // rdx
  int v58; // eax
  int v59; // r8d
  size_t *v60; // rsi
  __int64 v61; // [rsp+8h] [rbp-168h]
  __int64 v62; // [rsp+10h] [rbp-160h]
  int *v63; // [rsp+18h] [rbp-158h]
  __int64 v64; // [rsp+20h] [rbp-150h]
  __int64 v65; // [rsp+20h] [rbp-150h]
  int *v66; // [rsp+28h] [rbp-148h]
  int *v67; // [rsp+28h] [rbp-148h]
  _QWORD *v68; // [rsp+38h] [rbp-138h] BYREF
  __int64 v69[2]; // [rsp+40h] [rbp-130h] BYREF
  __int64 v70[2]; // [rsp+50h] [rbp-120h] BYREF
  unsigned __int64 v71; // [rsp+60h] [rbp-110h]
  unsigned __int64 v72; // [rsp+68h] [rbp-108h]
  __int64 v73; // [rsp+70h] [rbp-100h]
  unsigned __int64 *v74; // [rsp+78h] [rbp-F8h]
  _QWORD *v75; // [rsp+80h] [rbp-F0h]
  __int64 v76; // [rsp+88h] [rbp-E8h]
  __int64 v77; // [rsp+90h] [rbp-E0h]
  __int64 v78; // [rsp+98h] [rbp-D8h]
  int v79[52]; // [rsp+A0h] [rbp-D0h] BYREF

  v7 = a1;
  if ( !a2 || !sub_26C3E80(a1, a2) )
  {
    if ( !a3 )
      return;
    goto LABEL_4;
  }
  a5 = 0;
  if ( a3 )
  {
LABEL_4:
    if ( !unk_4F838D3 )
    {
      if ( (_BYTE)qword_4FF6D88 )
        a5 = 0;
      sub_26CF350(a3, a4, (_QWORD *)(a1 + 1296), a5);
      return;
    }
    v9 = *(_QWORD *)(a1 + 1512);
    v10 = *(_QWORD *)(v9 + 64);
    v11 = *(_QWORD **)(*(_QWORD *)(v9 + 56) + 8 * (a3 % v10));
    if ( v11 )
    {
      v12 = (_QWORD *)*v11;
      if ( a3 == *(_QWORD *)(*v11 + 8LL) )
      {
LABEL_15:
        v11 = (_QWORD *)*v11;
        if ( v11 )
          v11 = (_QWORD *)v11[2];
      }
      else
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( !*v12 )
            break;
          v11 = v12;
          if ( a3 % v10 != *(_QWORD *)(v13 + 8) % v10 )
            break;
          v12 = (_QWORD *)*v12;
          if ( a3 == *(_QWORD *)(v13 + 8) )
            goto LABEL_15;
        }
        v11 = 0;
      }
    }
    v68 = v11;
    v70[0] = 0;
    v70[1] = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    v76 = 0;
    v77 = 0;
    v78 = 0;
    sub_26C4970(v70, 0);
    v14 = v75;
    if ( v75 == (_QWORD *)(v77 - 8) )
    {
      sub_26C4A60((unsigned __int64 *)v70, &v68);
      v15 = v75;
    }
    else
    {
      if ( v75 )
      {
        *v75 = v68;
        v14 = v75;
      }
      v15 = v14 + 1;
      v75 = v14 + 1;
    }
    v16 = (__int64 *)v71;
    if ( (_QWORD *)v71 == v15 )
    {
LABEL_8:
      sub_26C2C00((unsigned __int64 *)v70);
      return;
    }
    while ( 1 )
    {
      v61 = *v16;
      if ( v16 == (__int64 *)(v73 - 8) )
      {
        j_j___libc_free_0(v72);
        v57 = *++v74 + 512;
        v72 = *v74;
        v73 = v57;
        v71 = v72;
      }
      else
      {
        v71 = (unsigned __int64)(v16 + 1);
      }
      v18 = sub_317E470(v61);
      v19 = v18;
      if ( !v18 || (!(_BYTE)qword_4FF6D88 || (*(_BYTE *)(v18 + 52) & 2) == 0) && sub_EF9210((_QWORD *)v18) < a5 )
      {
        v17 = v75;
        goto LABEL_24;
      }
      v20 = *(_QWORD *)(v19 + 24);
      v66 = *(int **)(v19 + 16);
      if ( v66 )
      {
        sub_C7D030(v79);
        sub_C7D280(v79, v66, v20);
        sub_C7D290(v79, v69);
        v20 = v69[0];
      }
      v21 = *(_QWORD *)(v7 + 1304);
      v22 = *(_QWORD **)(*(_QWORD *)(v7 + 1296) + 8 * (v20 % v21));
      if ( v22 )
      {
        v23 = (_QWORD *)*v22;
        if ( v20 == *(_QWORD *)(*v22 + 8LL) )
        {
LABEL_38:
          if ( *v22 )
          {
            v25 = *(_QWORD *)(*v22 + 16LL);
            if ( v25 )
            {
              if ( !sub_B2FC80(v25) )
                goto LABEL_42;
            }
          }
        }
        else
        {
          while ( 1 )
          {
            v24 = (_QWORD *)*v23;
            if ( !*v23 )
              break;
            v22 = v23;
            if ( v20 % v21 != v24[1] % v21 )
              break;
            v23 = (_QWORD *)*v23;
            if ( v20 == v24[1] )
              goto LABEL_38;
          }
        }
      }
      v69[0] = sub_26BA4C0(*(int **)(v19 + 16), *(_QWORD *)(v19 + 24));
      sub_D7AC80((__int64)v79, a4, v69);
LABEL_42:
      v62 = v19 + 80;
      v64 = *(_QWORD *)(v19 + 96);
      if ( v19 + 80 != v64 )
      {
        while ( 1 )
        {
          v26 = *(_QWORD **)(v64 + 64);
          if ( v26 )
            break;
LABEL_70:
          v64 = sub_220EF30(v64);
          if ( v62 == v64 )
            goto LABEL_71;
        }
        while ( 2 )
        {
          while ( 2 )
          {
            if ( v26[3] <= a5 )
            {
LABEL_45:
              v26 = (_QWORD *)*v26;
              if ( !v26 )
                goto LABEL_70;
              continue;
            }
            break;
          }
          v27 = v26[2];
          v67 = (int *)v26[1];
          if ( v67 )
          {
            sub_C7D030(v79);
            sub_C7D280(v79, v67, v27);
            sub_C7D290(v79, v69);
            v27 = v69[0];
            v28 = *(_QWORD *)(v7 + 1304);
            v29 = *(_QWORD **)(*(_QWORD *)(v7 + 1296) + 8 * (v69[0] % v28));
            v30 = v69[0] % v28;
            if ( v29 )
            {
LABEL_49:
              v31 = (_QWORD *)*v29;
              if ( *(_QWORD *)(*v29 + 8LL) == v27 )
              {
LABEL_53:
                if ( *v29 )
                {
                  v33 = *(_QWORD *)(*v29 + 16LL);
                  if ( v33 )
                  {
                    if ( !sub_B2FC80(v33) )
                      goto LABEL_45;
                  }
                }
              }
              else
              {
                while ( 1 )
                {
                  v32 = (_QWORD *)*v31;
                  if ( !*v31 )
                    break;
                  v29 = v31;
                  if ( v32[1] % v28 != v30 )
                    break;
                  v31 = (_QWORD *)*v31;
                  if ( v32[1] == v27 )
                    goto LABEL_53;
                }
              }
            }
            v27 = v26[2];
            if ( v26[1] )
            {
              v63 = (int *)v26[1];
              sub_C7D030(v79);
              sub_C7D280(v79, v63, v27);
              sub_C7D290(v79, v69);
              v27 = v69[0];
            }
            v34 = *(_DWORD *)(a4 + 24);
            v69[0] = v27;
            if ( v34 )
            {
LABEL_59:
              v35 = *(_QWORD *)(a4 + 8);
              v36 = (v34 - 1) & (((0xBF58476D1CE4E5B9LL * v27) >> 31) ^ (484763065 * v27));
              v37 = (size_t *)(v35 + 8LL * v36);
              v38 = *v37;
              if ( v27 == *v37 )
                goto LABEL_45;
              v39 = 1;
              v40 = 0;
              while ( v38 != -1 )
              {
                if ( v38 != -2 || v40 )
                  v37 = v40;
                v36 = (v34 - 1) & (v39 + v36);
                v38 = *(_QWORD *)(v35 + 8LL * v36);
                if ( v27 == v38 )
                  goto LABEL_45;
                ++v39;
                v40 = v37;
                v37 = (size_t *)(v35 + 8LL * v36);
              }
              v41 = *(_DWORD *)(a4 + 16);
              if ( !v40 )
                v40 = v37;
              ++*(_QWORD *)a4;
              v42 = v41 + 1;
              *(_QWORD *)v79 = v40;
              if ( 4 * (v41 + 1) < 3 * v34 )
              {
                if ( v34 - *(_DWORD *)(a4 + 20) - v42 <= v34 >> 3 )
                {
                  sub_A32210(a4, v34);
                  sub_A27FA0(a4, v69, v79);
                  v27 = v69[0];
                  v40 = *(size_t **)v79;
                  v42 = *(_DWORD *)(a4 + 16) + 1;
                }
                goto LABEL_67;
              }
LABEL_88:
              sub_A32210(a4, 2 * v34);
              v51 = *(_DWORD *)(a4 + 24);
              if ( v51 )
              {
                v27 = v69[0];
                v52 = v51 - 1;
                v53 = *(_QWORD *)(a4 + 8);
                v54 = v52 & (((0xBF58476D1CE4E5B9LL * v69[0]) >> 31) ^ (484763065 * LODWORD(v69[0])));
                v40 = (size_t *)(v53 + 8LL * v54);
                v55 = *v40;
                if ( *v40 == v69[0] )
                {
LABEL_90:
                  v56 = *(_DWORD *)(a4 + 16);
                  *(_QWORD *)v79 = v40;
                  v42 = v56 + 1;
                }
                else
                {
                  v59 = 1;
                  v60 = 0;
                  while ( v55 != -1 )
                  {
                    if ( !v60 && v55 == -2 )
                      v60 = v40;
                    v54 = v52 & (v59 + v54);
                    v40 = (size_t *)(v53 + 8LL * v54);
                    v55 = *v40;
                    if ( v69[0] == *v40 )
                      goto LABEL_90;
                    ++v59;
                  }
                  if ( !v60 )
                    v60 = v40;
                  v42 = *(_DWORD *)(a4 + 16) + 1;
                  *(_QWORD *)v79 = v60;
                  v40 = v60;
                }
              }
              else
              {
                v58 = *(_DWORD *)(a4 + 16);
                v27 = v69[0];
                v40 = 0;
                *(_QWORD *)v79 = 0;
                v42 = v58 + 1;
              }
LABEL_67:
              *(_DWORD *)(a4 + 16) = v42;
              if ( *v40 != -1 )
                --*(_DWORD *)(a4 + 20);
              *v40 = v27;
              v26 = (_QWORD *)*v26;
              if ( !v26 )
                goto LABEL_70;
              continue;
            }
          }
          else
          {
            v28 = *(_QWORD *)(v7 + 1304);
            v29 = *(_QWORD **)(*(_QWORD *)(v7 + 1296) + 8 * (v27 % v28));
            v30 = v27 % v28;
            if ( v29 )
              goto LABEL_49;
            v34 = *(_DWORD *)(a4 + 24);
            v69[0] = v26[2];
            if ( v34 )
              goto LABEL_59;
          }
          break;
        }
        ++*(_QWORD *)a4;
        *(_QWORD *)v79 = 0;
        goto LABEL_88;
      }
LABEL_71:
      v43 = sub_317E450(v61);
      v17 = v75;
      v44 = v43 + 8;
      if ( v43 + 8 != *(_QWORD *)(v43 + 24) )
      {
        v65 = v7;
        v45 = v75;
        v46 = a5;
        v47 = *(_QWORD *)(v43 + 24);
        do
        {
          while ( 1 )
          {
            *(_QWORD *)v79 = v47 + 40;
            if ( v45 != (_QWORD *)(v77 - 8) )
              break;
            sub_26C4A60((unsigned __int64 *)v70, v79);
            v45 = v75;
            v47 = sub_220EEE0(v47);
            if ( v44 == v47 )
              goto LABEL_78;
          }
          if ( v45 )
          {
            *v45 = v47 + 40;
            v45 = v75;
          }
          v75 = ++v45;
          v47 = sub_220EEE0(v47);
        }
        while ( v44 != v47 );
LABEL_78:
        a5 = v46;
        v17 = v45;
        v7 = v65;
      }
LABEL_24:
      v16 = (__int64 *)v71;
      if ( (_QWORD *)v71 == v17 )
        goto LABEL_8;
    }
  }
  v48 = *(_QWORD *)(a2 - 32);
  if ( v48 && !*(_BYTE *)v48 && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v48 + 24) )
    a3 = *(_QWORD *)(a2 - 32);
  v49 = sub_BD5D20(a3);
  v70[0] = sub_B2F650((__int64)v49, v50);
  sub_D7AC80((__int64)v79, a4, v70);
}
