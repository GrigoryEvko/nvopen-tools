// Function: sub_2E3C430
// Address: 0x2e3c430
//
void __fastcall sub_2E3C430(__int64 a1)
{
  _QWORD **v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // r9
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  __int64 *v7; // r14
  __int64 *v8; // rbx
  __int64 v9; // r13
  _QWORD **v10; // r12
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rcx
  __int64 **v14; // r14
  __int64 v15; // r12
  int v16; // esi
  __int64 v17; // r9
  __int64 v18; // rdi
  int v19; // esi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r11
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 *v28; // rbx
  __int64 *v29; // r13
  _QWORD *i; // rax
  __int64 v31; // r14
  __int64 v32; // r12
  _QWORD **v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rsi
  int v40; // ecx
  __int64 v41; // rdi
  int v42; // ecx
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // rax
  int v46; // edi
  __int64 v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // rax
  int v50; // edi
  unsigned int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // rax
  __int64 v56; // r14
  __int64 v57; // r13
  __int64 v58; // r8
  __int64 v59; // rax
  _DWORD *v60; // rdi
  bool v61; // al
  __int64 *v62; // r14
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rdi
  unsigned __int64 *v66; // rbx
  _QWORD **v67; // r12
  unsigned __int64 v68; // rdi
  int v69; // eax
  __int64 v70; // rcx
  int v71; // eax
  int v72; // r10d
  int v73; // eax
  int v74; // r10d
  int v75; // r10d
  __int64 v76; // [rsp+0h] [rbp-90h]
  __int64 v77; // [rsp+0h] [rbp-90h]
  __int64 v78; // [rsp+8h] [rbp-88h]
  __int64 v79; // [rsp+10h] [rbp-80h] BYREF
  __int64 v80; // [rsp+18h] [rbp-78h]
  unsigned __int64 v81; // [rsp+20h] [rbp-70h]
  _QWORD *v82; // [rsp+28h] [rbp-68h]
  _QWORD *v83; // [rsp+30h] [rbp-60h]
  unsigned __int64 *v84; // [rsp+38h] [rbp-58h]
  _QWORD *v85; // [rsp+40h] [rbp-50h]
  char *v86; // [rsp+48h] [rbp-48h]
  _QWORD *v87; // [rsp+50h] [rbp-40h]
  _QWORD **v88; // [rsp+58h] [rbp-38h]

  if ( *(_QWORD *)(*(_QWORD *)(a1 + 120) + 32LL) == *(_QWORD *)(*(_QWORD *)(a1 + 120) + 40LL) )
    return;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v80 = 8;
  v79 = sub_22077B0(0x40u);
  v2 = (_QWORD **)(v79 + 24);
  v3 = (_QWORD *)sub_22077B0(0x200u);
  v5 = *(_QWORD *)(a1 + 120);
  v84 = (unsigned __int64 *)(v79 + 24);
  v6 = v3 + 64;
  *(_QWORD *)(v79 + 24) = v3;
  v88 = v2;
  v86 = (char *)v3;
  v87 = v3 + 64;
  v85 = v3;
  v7 = *(__int64 **)(v5 + 40);
  v83 = v3 + 64;
  v8 = *(__int64 **)(v5 + 32);
  v82 = v3;
  v81 = (unsigned __int64)v3;
  if ( v8 != v7 )
  {
    while ( 1 )
    {
      v9 = *v8;
      if ( v3 == v6 - 2 )
      {
        v10 = v88;
        if ( 32 * ((((char *)v88 - (char *)v84) >> 3) - 1)
           + (((char *)v3 - v86) >> 4)
           + ((__int64)((__int64)v83 - v81) >> 4) == 0x7FFFFFFFFFFFFFFLL )
LABEL_80:
          sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
        if ( (unsigned __int64)(v80 - (((__int64)v88 - v79) >> 3)) <= 1 )
        {
          sub_2E3C2B0((unsigned __int64 *)&v79, 1u, 0);
          v10 = v88;
        }
        v10[1] = (_QWORD *)sub_22077B0(0x200u);
        v11 = v85;
        if ( v85 )
        {
          *v85 = v9;
          v11[1] = 0;
        }
        ++v8;
        v3 = *++v88;
        v12 = (__int64)(*v88 + 64);
        v86 = (char *)v3;
        v87 = (_QWORD *)v12;
        v85 = v3;
        if ( v7 == v8 )
        {
LABEL_15:
          v13 = (__int64 *)v81;
          if ( v3 == (_QWORD *)v81 )
            break;
          v76 = a1 + 88;
LABEL_17:
          v14 = (__int64 **)*v13;
          v15 = v13[1];
          if ( v13 == v83 - 2 )
          {
            j_j___libc_free_0((unsigned __int64)v82);
            v70 = *++v84 + 512;
            v82 = (_QWORD *)*v84;
            v83 = (_QWORD *)v70;
            v81 = (unsigned __int64)v82;
          }
          else
          {
            v81 += 16LL;
          }
          v16 = *(_DWORD *)(a1 + 184);
          v17 = *(_QWORD *)(a1 + 168);
          v18 = *v14[4];
          if ( v16 )
          {
            v19 = v16 - 1;
            v20 = v19 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v21 = (__int64 *)(v17 + 16LL * v20);
            v22 = *v21;
            if ( v18 == *v21 )
            {
LABEL_21:
              v23 = *((unsigned int *)v21 + 2);
              v24 = 24 * v23;
              goto LABEL_22;
            }
            v69 = 1;
            while ( v22 != -4096 )
            {
              v74 = v69 + 1;
              v20 = v19 & (v69 + v20);
              v21 = (__int64 *)(v17 + 16LL * v20);
              v22 = *v21;
              if ( v18 == *v21 )
                goto LABEL_21;
              v69 = v74;
            }
          }
          v24 = 0x17FFFFFFE8LL;
          LODWORD(v23) = -1;
LABEL_22:
          v25 = sub_22077B0(0xC0u);
          *(_QWORD *)(v25 + 16) = v15;
          *(_BYTE *)(v25 + 24) = 0;
          *(_DWORD *)(v25 + 28) = 1;
          *(_QWORD *)(v25 + 32) = v25 + 48;
          *(_QWORD *)(v25 + 40) = 0x400000000LL;
          *(_QWORD *)(v25 + 112) = v25 + 128;
          *(_QWORD *)(v25 + 120) = 0x400000001LL;
          *(_QWORD *)(v25 + 144) = v25 + 160;
          *(_QWORD *)(v25 + 152) = 0x100000001LL;
          *(_DWORD *)(v25 + 128) = v23;
          *(_WORD *)(v25 + 184) = 0;
          *(_QWORD *)(v25 + 160) = 0;
          *(_QWORD *)(v25 + 168) = 0;
          *(_QWORD *)(v25 + 176) = 0;
          sub_2208C80((_QWORD *)v25, v76);
          v26 = *(_QWORD *)(a1 + 96);
          v27 = *(_QWORD *)(a1 + 64);
          ++*(_QWORD *)(a1 + 104);
          *(_QWORD *)(v27 + v24 + 8) = v26 + 16;
          v28 = v14[1];
          v29 = v14[2];
          for ( i = v85; v29 != v28; v85 = i )
          {
            while ( 1 )
            {
              v31 = *v28;
              v32 = *(_QWORD *)(a1 + 96) + 16LL;
              if ( i == v87 - 2 )
                break;
              if ( i )
              {
                *i = v31;
                i[1] = v32;
                i = v85;
              }
              i += 2;
              ++v28;
              v85 = i;
              if ( v29 == v28 )
                goto LABEL_34;
            }
            v33 = v88;
            if ( 32 * ((((char *)v88 - (char *)v84) >> 3) - 1)
               + (((char *)i - v86) >> 4)
               + ((__int64)((__int64)v83 - v81) >> 4) == 0x7FFFFFFFFFFFFFFLL )
              goto LABEL_80;
            if ( (unsigned __int64)(v80 - (((__int64)v88 - v79) >> 3)) <= 1 )
            {
              sub_2E3C2B0((unsigned __int64 *)&v79, 1u, 0);
              v33 = v88;
            }
            v33[1] = (_QWORD *)sub_22077B0(0x200u);
            v34 = v85;
            if ( v85 )
            {
              *v85 = v31;
              v34[1] = v32;
            }
            ++v28;
            i = *++v88;
            v35 = (__int64)(*v88 + 64);
            v86 = (char *)i;
            v87 = (_QWORD *)v35;
          }
LABEL_34:
          v13 = (__int64 *)v81;
          if ( (_QWORD *)v81 == i )
            break;
          goto LABEL_17;
        }
      }
      else
      {
        if ( v3 )
        {
          *v3 = v9;
          v3[1] = 0;
          v3 = v85;
        }
        v3 += 2;
        ++v8;
        v85 = v3;
        if ( v7 == v8 )
          goto LABEL_15;
      }
      v6 = v87;
    }
  }
  v36 = *(_QWORD *)(a1 + 136);
  v37 = 0;
  if ( *(_QWORD *)(a1 + 144) != v36 )
  {
    while ( 1 )
    {
      v56 = *(_QWORD *)(a1 + 64);
      v57 = v56 + 24 * v37;
      v58 = *(_QWORD *)(v57 + 8);
      if ( !v58 )
        break;
      v59 = *(unsigned int *)(v58 + 12);
      v60 = *(_DWORD **)(v58 + 96);
      if ( (unsigned int)v59 <= 1 )
      {
        if ( *(_DWORD *)v57 != *v60 )
          break;
      }
      else
      {
        v77 = *(_QWORD *)(v57 + 8);
        v78 = v36;
        v61 = sub_FDC990(v60, &v60[v59], (_DWORD *)(v56 + 24 * v37));
        v36 = v78;
        v58 = v77;
        if ( !v61 )
          break;
      }
      v62 = *(__int64 **)v58;
      if ( *(_QWORD *)v58 )
      {
        v63 = *((unsigned int *)v62 + 3);
        if ( (unsigned int)v63 <= 1
          || !sub_FDC990((_DWORD *)v62[12], (_DWORD *)(v62[12] + 4 * v63), (_DWORD *)v57)
          || (v62 = (__int64 *)*v62) != 0 )
        {
          v64 = *((unsigned int *)v62 + 26);
          if ( v64 + 1 > (unsigned __int64)*((unsigned int *)v62 + 27) )
          {
            sub_C8D5F0((__int64)(v62 + 12), v62 + 14, v64 + 1, 4u, v58, v4);
            v64 = *((unsigned int *)v62 + 26);
          }
          *(_DWORD *)(v62[12] + 4 * v64) = v37;
          ++*((_DWORD *)v62 + 26);
        }
      }
LABEL_47:
      v36 = *(_QWORD *)(a1 + 136);
      if ( ++v37 >= (unsigned __int64)((*(_QWORD *)(a1 + 144) - v36) >> 3) )
        goto LABEL_56;
    }
    v38 = *(_QWORD *)(a1 + 120);
    v39 = *(_QWORD *)(v36 + 8 * v37);
    v40 = *(_DWORD *)(v38 + 24);
    v41 = *(_QWORD *)(v38 + 8);
    if ( v40 )
    {
      v42 = v40 - 1;
      v43 = v42 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v44 = (__int64 *)(v41 + 16LL * v43);
      v4 = *v44;
      if ( v39 == *v44 )
      {
LABEL_40:
        v45 = v44[1];
        if ( v45 )
        {
          v46 = *(_DWORD *)(a1 + 184);
          v47 = *(_QWORD *)(a1 + 168);
          v48 = **(_QWORD **)(v45 + 32);
          v49 = 0x17FFFFFFE8LL;
          if ( v46 )
          {
            v50 = v46 - 1;
            v51 = v50 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v52 = v47 + 16LL * v51;
            v4 = *(_QWORD *)v52;
            if ( v48 == *(_QWORD *)v52 )
            {
LABEL_43:
              v49 = 24LL * *(unsigned int *)(v52 + 8);
            }
            else
            {
              v73 = 1;
              while ( v4 != -4096 )
              {
                v75 = v73 + 1;
                v51 = v50 & (v73 + v51);
                v52 = v47 + 16LL * v51;
                v4 = *(_QWORD *)v52;
                if ( v48 == *(_QWORD *)v52 )
                  goto LABEL_43;
                v73 = v75;
              }
              v49 = 0x17FFFFFFE8LL;
            }
          }
          v53 = v56 + v49;
          *(_QWORD *)(v57 + 8) = *(_QWORD *)(v53 + 8);
          v54 = *(_QWORD *)(v53 + 8);
          v55 = *(unsigned int *)(v54 + 104);
          if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v54 + 108) )
          {
            sub_C8D5F0(v54 + 96, (const void *)(v54 + 112), v55 + 1, 4u, v58, v4);
            v55 = *(unsigned int *)(v54 + 104);
          }
          *(_DWORD *)(*(_QWORD *)(v54 + 96) + 4 * v55) = v37;
          ++*(_DWORD *)(v54 + 104);
        }
      }
      else
      {
        v71 = 1;
        while ( v4 != -4096 )
        {
          v72 = v71 + 1;
          v43 = v42 & (v71 + v43);
          v44 = (__int64 *)(v41 + 16LL * v43);
          v4 = *v44;
          if ( v39 == *v44 )
            goto LABEL_40;
          v71 = v72;
        }
      }
    }
    goto LABEL_47;
  }
LABEL_56:
  v65 = v79;
  if ( v79 )
  {
    v66 = v84;
    v67 = v88 + 1;
    if ( v88 + 1 > (_QWORD **)v84 )
    {
      do
      {
        v68 = *v66++;
        j_j___libc_free_0(v68);
      }
      while ( v67 > (_QWORD **)v66 );
      v65 = v79;
    }
    j_j___libc_free_0(v65);
  }
}
