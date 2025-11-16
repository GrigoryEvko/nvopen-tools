// Function: sub_2965DD0
// Address: 0x2965dd0
//
void __fastcall sub_2965DD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        char a7,
        void *src,
        __int64 a9)
{
  __int64 v9; // r15
  __int64 v10; // r13
  const void *v11; // r8
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  bool v25; // zf
  __int64 v26; // r12
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned __int64 v30; // rdx
  __int64 v31; // r14
  unsigned int v32; // ebx
  __int64 v33; // rbx
  __int64 v35; // r10
  int v36; // esi
  unsigned int v37; // edx
  const char **v38; // rax
  const char *v39; // r11
  __int64 v40; // rdx
  __int64 v41; // r9
  const char **v42; // r8
  const char *v43; // rcx
  char v44; // di
  unsigned int v45; // esi
  __int64 v46; // rbx
  const char *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // r9
  const char **v51; // rdx
  const char *v52; // rcx
  __int64 v53; // rax
  __int64 v54; // r12
  unsigned int v55; // eax
  int v56; // eax
  unsigned int v57; // esi
  unsigned int v58; // ecx
  __int64 v59; // rcx
  const char **v60; // rax
  __int64 v61; // rdx
  unsigned int v62; // eax
  int v63; // edx
  unsigned int v64; // edi
  const char **v65; // rax
  const char **v66; // r13
  int v70; // [rsp+30h] [rbp-70h]
  char v71; // [rsp+38h] [rbp-68h]
  const char **v73; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v74[2]; // [rsp+50h] [rbp-50h] BYREF
  const char *v75; // [rsp+60h] [rbp-40h] BYREF
  __int64 v76; // [rsp+68h] [rbp-38h]

  v9 = a2;
  v10 = a1;
  v11 = src;
  v71 = a6;
  if ( a9 )
  {
    v25 = *(_BYTE *)(a2 + 25) == 0;
    v26 = *(_QWORD *)a2;
    v74[0] = src;
    v74[1] = a9;
    if ( v25 )
    {
      sub_F76FB0(v74, v26, a3, a4, (__int64)src, a6);
      goto LABEL_2;
    }
    v27 = 8 * a9;
    if ( 8 * a9 )
    {
      v28 = *(unsigned int *)(v26 + 88);
      v29 = v27 >> 3;
      v30 = v28 + (v27 >> 3);
      v31 = v28;
      if ( v30 > *(unsigned int *)(v26 + 92) )
      {
        sub_C8D5F0(v26 + 80, (const void *)(v26 + 96), v30, 8u, (__int64)src, v27);
        v28 = *(unsigned int *)(v26 + 88);
        v11 = src;
        v27 = 8 * a9;
      }
      memcpy((void *)(*(_QWORD *)(v26 + 80) + 8 * v28), v11, v27);
      v32 = *(_DWORD *)(v26 + 88) + v29;
      *(_DWORD *)(v26 + 88) = v32;
      v33 = v32 - 1LL;
      if ( v31 <= v33 )
      {
        while ( 1 )
        {
          v41 = *(_QWORD *)(v26 + 80);
          v42 = (const char **)(v41 + 8 * v33);
          v43 = *v42;
          v76 = v33;
          v75 = v43;
          v44 = *(_BYTE *)(v26 + 8) & 1;
          if ( v44 )
          {
            v35 = v26 + 16;
            v36 = 3;
          }
          else
          {
            v45 = *(_DWORD *)(v26 + 24);
            v35 = *(_QWORD *)(v26 + 16);
            if ( !v45 )
            {
              v73 = 0;
              v62 = *(_DWORD *)(v26 + 8);
              ++*(_QWORD *)v26;
              v63 = (v62 >> 1) + 1;
              goto LABEL_41;
            }
            v36 = v45 - 1;
          }
          v37 = v36 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v38 = (const char **)(v35 + 16LL * v37);
          v39 = *v38;
          if ( v43 != *v38 )
            break;
LABEL_19:
          v40 = (__int64)v38[1];
          if ( v31 > v40 )
          {
            *(_QWORD *)(v41 + 8 * v40) = 0;
            v38[1] = (const char *)v33--;
            if ( v31 > v33 )
            {
LABEL_39:
              v10 = a1;
              v9 = a2;
              goto LABEL_2;
            }
          }
          else
          {
            *v42 = 0;
LABEL_21:
            if ( v31 > --v33 )
              goto LABEL_39;
          }
        }
        v70 = 1;
        v66 = 0;
        while ( v39 != (const char *)-4096LL )
        {
          if ( !v66 && v39 == (const char *)-8192LL )
            v66 = v38;
          v37 = v36 & (v70 + v37);
          v38 = (const char **)(v35 + 16LL * v37);
          v39 = *v38;
          if ( v43 == *v38 )
          {
            v42 = (const char **)(v41 + 8 * v33);
            goto LABEL_19;
          }
          ++v70;
        }
        if ( !v66 )
          v66 = v38;
        v73 = v66;
        v62 = *(_DWORD *)(v26 + 8);
        ++*(_QWORD *)v26;
        v63 = (v62 >> 1) + 1;
        if ( v44 )
        {
          v64 = 12;
          v45 = 4;
          goto LABEL_42;
        }
        v45 = *(_DWORD *)(v26 + 24);
LABEL_41:
        v64 = 3 * v45;
LABEL_42:
        if ( v64 <= 4 * v63 )
        {
          v45 *= 2;
        }
        else if ( v45 - *(_DWORD *)(v26 + 12) - v63 > v45 >> 3 )
        {
LABEL_44:
          *(_DWORD *)(v26 + 8) = (2 * (v62 >> 1) + 2) | v62 & 1;
          v65 = v73;
          if ( *v73 != (const char *)-4096LL )
            --*(_DWORD *)(v26 + 12);
          *v65 = v75;
          v65[1] = (const char *)v76;
          goto LABEL_21;
        }
        sub_F76580(v26, v45);
        sub_295D920(v26, (__int64 *)&v75, &v73);
        v62 = *(_DWORD *)(v26 + 8);
        goto LABEL_44;
      }
    }
  }
LABEL_2:
  if ( a5 )
  {
    if ( v71 )
    {
      v12 = (__int64 *)sub_AA48A0(**(_QWORD **)(v10 + 32));
      v75 = (const char *)sub_B9B140(v12, "llvm.loop.unswitch.partial.disable", 0x22u);
      v24 = sub_B9C770(v12, (__int64 *)&v75, (__int64 *)1, 0, 1);
      v76 = 26;
      v74[0] = v24;
      v75 = "llvm.loop.unswitch.partial";
      goto LABEL_6;
    }
    if ( a7 )
    {
      v12 = (__int64 *)sub_AA48A0(**(_QWORD **)(v10 + 32));
      v75 = (const char *)sub_B9B140(v12, "llvm.loop.unswitch.injection.disable", 0x24u);
      v13 = sub_B9C770(v12, (__int64 *)&v75, (__int64 *)1, 0, 1);
      v76 = 28;
      v74[0] = v13;
      v75 = "llvm.loop.unswitch.injection";
LABEL_6:
      v18 = sub_D49300(v10, (__int64)&v75, v14, v15, v16, v17);
      v19 = sub_D4A520(v12, v18, (__int64)&v75, 1, (__int64)v74, 1);
      sub_D49440(v10, (__int64)v19, v20, v21, v22, v23);
      return;
    }
    v46 = *(_QWORD *)v9;
    v47 = *(const char **)(v9 + 16);
    *(_BYTE *)(v9 + 24) = 1;
    v48 = *(unsigned int *)(v46 + 88);
    v75 = v47;
    v76 = v48;
    if ( (unsigned __int8)sub_295D920(v46, (__int64 *)&v75, &v73) )
    {
      v51 = v73;
      v52 = v73[1];
      if ( v52 != (const char *)(*(unsigned int *)(v46 + 88) - 1LL) )
      {
        *(_QWORD *)(*(_QWORD *)(v46 + 80) + 8LL * (_QWORD)v52) = 0;
        v51[1] = (const char *)*(unsigned int *)(v46 + 88);
        v53 = *(unsigned int *)(v46 + 88);
        v54 = *(_QWORD *)(v9 + 16);
        if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(v46 + 92) )
        {
          sub_C8D5F0(v46 + 80, (const void *)(v46 + 96), v53 + 1, 8u, v49, v50);
          v53 = *(unsigned int *)(v46 + 88);
        }
        *(_QWORD *)(*(_QWORD *)(v46 + 80) + 8 * v53) = v54;
        ++*(_DWORD *)(v46 + 88);
      }
      return;
    }
    v74[0] = v73;
    v55 = *(_DWORD *)(v46 + 8);
    ++*(_QWORD *)v46;
    v56 = (v55 >> 1) + 1;
    if ( (*(_BYTE *)(v46 + 8) & 1) != 0 )
    {
      v58 = 12;
      v57 = 4;
    }
    else
    {
      v57 = *(_DWORD *)(v46 + 24);
      v58 = 3 * v57;
    }
    if ( v58 <= 4 * v56 )
    {
      v57 *= 2;
    }
    else
    {
      v59 = v57 - (v56 + *(_DWORD *)(v46 + 12));
      if ( (unsigned int)v59 > v57 >> 3 )
      {
LABEL_35:
        *(_DWORD *)(v46 + 8) = *(_DWORD *)(v46 + 8) & 1 | (2 * v56);
        v60 = (const char **)v74[0];
        if ( *(_QWORD *)v74[0] != -4096 )
          --*(_DWORD *)(v46 + 12);
        *v60 = v75;
        v61 = v76;
        v60[1] = (const char *)v76;
        sub_295C830(v46 + 80, *(_QWORD *)(v9 + 16), v61, v59, v49, v50);
        return;
      }
    }
    sub_F76580(v46, v57);
    sub_295D920(v46, (__int64 *)&v75, v74);
    v56 = (*(_DWORD *)(v46 + 8) >> 1) + 1;
    goto LABEL_35;
  }
  sub_22D0060(*(_QWORD *)(v9 + 8), v10, a3, a4);
  if ( v10 == *(_QWORD *)(v9 + 16) )
    *(_BYTE *)(v9 + 24) = 1;
}
