// Function: sub_3588750
// Address: 0x3588750
//
_QWORD *__fastcall sub_3588750(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned int v9; // esi
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 *v12; // rdi
  int v13; // r11d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 *v19; // r12
  char v20; // al
  char v21; // r8
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // r9
  int v25; // edx
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // r10
  __int64 v30; // rcx
  unsigned int v31; // esi
  __int64 *v32; // rax
  __int64 v33; // r11
  bool v34; // al
  unsigned int v35; // esi
  __int64 v36; // r15
  __int64 v37; // r9
  int v38; // r11d
  __int64 *v39; // rdi
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // r8
  __int64 *v43; // rax
  __int64 *v44; // rdx
  __int64 *v45; // rcx
  unsigned int v46; // edi
  __int64 *i; // rax
  __int64 v48; // rsi
  unsigned __int64 v49; // rax
  _QWORD *result; // rax
  int v51; // eax
  int v52; // eax
  __int64 v53; // rbx
  int v54; // eax
  int v55; // eax
  int v56; // eax
  int v57; // edx
  int v58; // r10d
  int v59; // r10d
  unsigned int v60; // edx
  __int64 v61; // rsi
  __int64 *v62; // r11
  int v63; // r10d
  int v64; // r10d
  unsigned int v65; // edx
  __int64 v66; // rsi
  int v67; // r15d
  int v68; // esi
  __int64 v69; // [rsp+8h] [rbp-68h]
  unsigned __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+28h] [rbp-48h] BYREF
  __int64 v72; // [rsp+30h] [rbp-40h] BYREF
  __int64 v73[7]; // [rsp+38h] [rbp-38h] BYREF

  v71 = a2;
  v9 = *(_DWORD *)(a1 + 992);
  v69 = a1 + 968;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 968);
    v73[0] = 0;
LABEL_72:
    v9 *= 2;
    goto LABEL_73;
  }
  v10 = v71;
  v11 = *(_QWORD *)(a1 + 976);
  v12 = 0;
  v13 = 1;
  v14 = (v9 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v71 == *v15 )
  {
LABEL_3:
    v17 = v15[1];
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( !v12 && v16 == -8192 )
      v12 = v15;
    v14 = (v9 - 1) & (v13 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v71 == *v15 )
      goto LABEL_3;
    ++v13;
  }
  if ( !v12 )
    v12 = v15;
  v56 = *(_DWORD *)(a1 + 984);
  ++*(_QWORD *)(a1 + 968);
  v57 = v56 + 1;
  v73[0] = (__int64)v12;
  if ( 4 * (v56 + 1) >= 3 * v9 )
    goto LABEL_72;
  if ( v9 - *(_DWORD *)(a1 + 988) - v57 <= v9 >> 3 )
  {
LABEL_73:
    sub_35793B0(v69, v9);
    sub_3585490(v69, &v71, v73);
    v10 = v71;
    v12 = (__int64 *)v73[0];
    v57 = *(_DWORD *)(a1 + 984) + 1;
  }
  *(_DWORD *)(a1 + 984) = v57;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a1 + 988);
  *v12 = v10;
  v17 = 0;
  v12[1] = 0;
LABEL_4:
  v72 = v17;
  v18 = a1 + 40;
  v19 = &a3[a4];
  v70 = *sub_3588500(a1 + 40, &v72);
  if ( v19 == a3 )
    goto LABEL_29;
  do
  {
    v73[0] = *a3;
    sub_2EB3EB0(a5, v73[0], v71);
    v21 = v20;
    v22 = *(_QWORD *)(a1 + 1016);
    v23 = *(_DWORD *)(v22 + 24);
    v24 = *(_QWORD *)(v22 + 8);
    if ( v23 )
    {
      v25 = v23 - 1;
      v26 = v25 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
      v27 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v27;
      if ( v71 == *v27 )
      {
LABEL_7:
        v29 = v27[1];
      }
      else
      {
        v52 = 1;
        while ( v28 != -4096 )
        {
          v68 = v52 + 1;
          v26 = v25 & (v52 + v26);
          v27 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v27;
          if ( v71 == *v27 )
            goto LABEL_7;
          v52 = v68;
        }
        v29 = 0;
      }
      v30 = v73[0];
      v31 = v25 & ((LODWORD(v73[0]) >> 9) ^ (LODWORD(v73[0]) >> 4));
      v32 = (__int64 *)(v24 + 16LL * v31);
      v33 = *v32;
      if ( *v32 == v73[0] )
      {
LABEL_9:
        v34 = v32[1] == v29;
      }
      else
      {
        v51 = 1;
        while ( v33 != -4096 )
        {
          v67 = v51 + 1;
          v31 = v25 & (v51 + v31);
          v32 = (__int64 *)(v24 + 16LL * v31);
          v33 = *v32;
          if ( *v32 == v73[0] )
            goto LABEL_9;
          v51 = v67;
        }
        v34 = v29 == 0;
      }
    }
    else
    {
      v30 = v73[0];
      v34 = 1;
    }
    if ( v71 != v30 && v21 && v34 )
    {
      v35 = *(_DWORD *)(a1 + 992);
      v36 = v72;
      if ( v35 )
      {
        v37 = *(_QWORD *)(a1 + 976);
        v38 = 1;
        v39 = 0;
        v40 = (v35 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v41 = (__int64 *)(v37 + 16LL * v40);
        v42 = *v41;
        if ( *v41 == v30 )
        {
LABEL_15:
          v43 = v41 + 1;
          goto LABEL_16;
        }
        while ( v42 != -4096 )
        {
          if ( !v39 && v42 == -8192 )
            v39 = v41;
          v40 = (v35 - 1) & (v38 + v40);
          v41 = (__int64 *)(v37 + 16LL * v40);
          v42 = *v41;
          if ( *v41 == v30 )
            goto LABEL_15;
          ++v38;
        }
        if ( !v39 )
          v39 = v41;
        v54 = *(_DWORD *)(a1 + 984);
        ++*(_QWORD *)(a1 + 968);
        v55 = v54 + 1;
        if ( 4 * v55 < 3 * v35 )
        {
          v42 = v35 >> 3;
          if ( v35 - *(_DWORD *)(a1 + 988) - v55 > (unsigned int)v42 )
            goto LABEL_55;
          sub_35793B0(v69, v35);
          v63 = *(_DWORD *)(a1 + 992);
          if ( !v63 )
          {
LABEL_99:
            ++*(_DWORD *)(a1 + 984);
            BUG();
          }
          v30 = v73[0];
          v64 = v63 - 1;
          v42 = *(_QWORD *)(a1 + 976);
          v62 = 0;
          v37 = 1;
          v65 = v64 & ((LODWORD(v73[0]) >> 9) ^ (LODWORD(v73[0]) >> 4));
          v55 = *(_DWORD *)(a1 + 984) + 1;
          v39 = (__int64 *)(v42 + 16LL * v65);
          v66 = *v39;
          if ( *v39 == v73[0] )
            goto LABEL_55;
          while ( v66 != -4096 )
          {
            if ( !v62 && v66 == -8192 )
              v62 = v39;
            v65 = v64 & (v37 + v65);
            v39 = (__int64 *)(v42 + 16LL * v65);
            v66 = *v39;
            if ( v73[0] == *v39 )
              goto LABEL_55;
            v37 = (unsigned int)(v37 + 1);
          }
          goto LABEL_87;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 968);
      }
      sub_35793B0(v69, 2 * v35);
      v58 = *(_DWORD *)(a1 + 992);
      if ( !v58 )
        goto LABEL_99;
      v30 = v73[0];
      v59 = v58 - 1;
      v42 = *(_QWORD *)(a1 + 976);
      v60 = v59 & ((LODWORD(v73[0]) >> 9) ^ (LODWORD(v73[0]) >> 4));
      v55 = *(_DWORD *)(a1 + 984) + 1;
      v39 = (__int64 *)(v42 + 16LL * v60);
      v61 = *v39;
      if ( *v39 == v73[0] )
        goto LABEL_55;
      v37 = 1;
      v62 = 0;
      while ( v61 != -4096 )
      {
        if ( !v62 && v61 == -8192 )
          v62 = v39;
        v60 = v59 & (v37 + v60);
        v39 = (__int64 *)(v42 + 16LL * v60);
        v61 = *v39;
        if ( v73[0] == *v39 )
          goto LABEL_55;
        v37 = (unsigned int)(v37 + 1);
      }
LABEL_87:
      if ( v62 )
        v39 = v62;
LABEL_55:
      *(_DWORD *)(a1 + 984) = v55;
      if ( *v39 != -4096 )
        --*(_DWORD *)(a1 + 988);
      *v39 = v30;
      v43 = v39 + 1;
      v39[1] = 0;
LABEL_16:
      *v43 = v36;
      if ( *(_BYTE *)(a1 + 132) )
      {
        v44 = *(__int64 **)(a1 + 112);
        v45 = &v44[*(unsigned int *)(a1 + 124)];
        v46 = *(_DWORD *)(a1 + 124);
        for ( i = v44; v45 != i; ++i )
        {
          if ( v73[0] == *i )
          {
            v48 = v72;
            goto LABEL_23;
          }
        }
      }
      else if ( sub_C8CA60(a1 + 104, v73[0]) )
      {
        v48 = v72;
        if ( !*(_BYTE *)(a1 + 132) )
          goto LABEL_40;
        v44 = *(__int64 **)(a1 + 112);
        v45 = &v44[*(unsigned int *)(a1 + 124)];
        v46 = *(_DWORD *)(a1 + 124);
        if ( v45 != v44 )
        {
LABEL_23:
          while ( v48 != *v44 )
          {
            if ( v45 == ++v44 )
              goto LABEL_43;
          }
          goto LABEL_24;
        }
LABEL_43:
        if ( *(_DWORD *)(a1 + 120) > v46 )
        {
          *(_DWORD *)(a1 + 124) = v46 + 1;
          *v45 = v48;
          ++*(_QWORD *)(a1 + 104);
        }
        else
        {
LABEL_40:
          sub_C8CC70(a1 + 104, v48, (__int64)v44, (__int64)v45, v42, v37);
        }
      }
LABEL_24:
      v49 = *sub_3588500(a1 + 40, v73);
      if ( v70 >= v49 )
        v49 = v70;
      v70 = v49;
    }
    ++a3;
  }
  while ( v19 != a3 );
  v18 = a1 + 40;
LABEL_29:
  if ( v72 == *(_QWORD *)(*(_QWORD *)(v72 + 32) + 328LL) )
  {
    v53 = *(_QWORD *)(*(_QWORD *)(a1 + 1200) + 64LL);
    result = sub_3588500(v18, &v72);
    *result = v53 + 1;
  }
  else
  {
    result = sub_3588500(v18, &v72);
    *result = v70;
  }
  return result;
}
