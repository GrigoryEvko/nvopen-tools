// Function: sub_32B0630
// Address: 0x32b0630
//
__int64 __fastcall sub_32B0630(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 *v11; // rax
  char v12; // dl
  __int64 v13; // rdx
  __int64 *v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 *v20; // rbx
  __int64 v21; // rax
  __int64 *v22; // r14
  __int64 v23; // r15
  __int64 i; // rbx
  unsigned int v25; // r12d
  unsigned int v26; // esi
  __int64 v27; // rbx
  __int64 v28; // r9
  __int64 v29; // r10
  int v30; // ecx
  __int64 *v31; // rdx
  unsigned int v32; // r8d
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // rdx
  __int64 v36; // rax
  int v38; // edx
  int v39; // r8d
  int v40; // r8d
  __int64 v41; // r9
  int v42; // ecx
  unsigned int v43; // r12d
  __int64 *v44; // rdi
  __int64 v45; // rsi
  int v46; // r8d
  int v47; // r8d
  __int64 v48; // r9
  unsigned int v49; // ecx
  __int64 v50; // rdi
  int v51; // esi
  __int64 *v52; // r10
  unsigned int v55; // [rsp+1Ch] [rbp-1C4h]
  __int64 v57; // [rsp+38h] [rbp-1A8h]
  __int64 *v58; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v59; // [rsp+48h] [rbp-198h]
  _QWORD v60[8]; // [rsp+50h] [rbp-190h] BYREF
  __int64 v61; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v62; // [rsp+98h] [rbp-148h]
  __int64 v63; // [rsp+A0h] [rbp-140h]
  int v64; // [rsp+A8h] [rbp-138h]
  unsigned __int8 v65; // [rsp+ACh] [rbp-134h]
  char v66; // [rsp+B0h] [rbp-130h] BYREF

  v7 = v60;
  v61 = 0;
  v63 = 32;
  v64 = 0;
  v65 = 1;
  v58 = v60;
  v62 = (__int64 *)&v66;
  v60[0] = a4;
  v8 = 1;
  v59 = 0x800000001LL;
  LODWORD(v9) = 1;
  while ( 1 )
  {
    v10 = v7[(unsigned int)v9 - 1];
    LODWORD(v59) = v9 - 1;
    if ( !(_BYTE)v8 )
      goto LABEL_10;
    v11 = v62;
    v8 = HIDWORD(v63);
    v7 = &v62[HIDWORD(v63)];
    if ( v62 == v7 )
    {
LABEL_17:
      if ( HIDWORD(v63) >= (unsigned int)v63 )
      {
LABEL_10:
        sub_C8CC70((__int64)&v61, v10, (__int64)v7, v8, a5, a6);
        v9 = (unsigned int)v59;
        if ( !v12 )
          goto LABEL_8;
      }
      else
      {
        ++HIDWORD(v63);
        *v7 = v10;
        v9 = (unsigned int)v59;
        ++v61;
      }
      if ( *(_DWORD *)(v10 + 24) == 2 )
      {
        v13 = *(_QWORD *)(v10 + 40);
        a5 = v13 + 40LL * *(unsigned int *)(v10 + 64);
        if ( a5 != v13 )
        {
          v14 = *(__int64 **)(v10 + 40);
          do
          {
            v15 = *v14;
            if ( v9 + 1 > (unsigned __int64)HIDWORD(v59) )
            {
              v57 = a5;
              sub_C8D5F0((__int64)&v58, v60, v9 + 1, 8u, a5, a6);
              v9 = (unsigned int)v59;
              a5 = v57;
            }
            v14 += 5;
            v58[v9] = v15;
            v9 = (unsigned int)(v59 + 1);
            LODWORD(v59) = v59 + 1;
          }
          while ( (__int64 *)a5 != v14 );
        }
      }
      goto LABEL_8;
    }
    while ( v10 != *v11 )
    {
      if ( v7 == ++v11 )
        goto LABEL_17;
    }
    LODWORD(v9) = v59;
LABEL_8:
    if ( !(_DWORD)v9 )
      break;
    v7 = v58;
    v8 = v65;
  }
  if ( !a3 )
  {
LABEL_38:
    v25 = 1;
    goto LABEL_39;
  }
  v16 = *a2;
  v55 = HIDWORD(v63) + 1024 - v64;
  v17 = 0;
  do
  {
    v18 = *(_QWORD *)(v16 + v17);
    v19 = *(_QWORD *)(v18 + 40);
    v20 = (__int64 *)(v19 + 40LL * *(unsigned int *)(v18 + 64));
    if ( (__int64 *)v19 != v20 )
    {
      v21 = (unsigned int)v59;
      v22 = *(__int64 **)(v18 + 40);
      do
      {
        v23 = *v22;
        if ( v21 + 1 > (unsigned __int64)HIDWORD(v59) )
        {
          sub_C8D5F0((__int64)&v58, v60, v21 + 1, 8u, v19, a6);
          v21 = (unsigned int)v59;
        }
        v22 += 5;
        v58[v21] = v23;
        v21 = (unsigned int)(v59 + 1);
        LODWORD(v59) = v59 + 1;
      }
      while ( v20 != v22 );
      v16 = *a2;
    }
    v17 += 16;
  }
  while ( 16LL * a3 != v17 );
  for ( i = 0; !(unsigned __int8)sub_3285B00(*(_QWORD *)(v16 + i), (__int64)&v61, (__int64)&v58, v55, 0, a6); i += 16 )
  {
    if ( i == 16LL * (a3 - 1) )
      goto LABEL_38;
    v16 = *a2;
  }
  v25 = 0;
  if ( v55 <= HIDWORD(v63) - v64 )
  {
    v26 = *(_DWORD *)(a1 + 896);
    v27 = *(_QWORD *)(*a2 + i);
    v28 = a1 + 872;
    if ( v26 )
    {
      v29 = *(_QWORD *)(a1 + 880);
      v30 = 1;
      v31 = 0;
      v32 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v33 = (__int64 *)(v29 + 24LL * v32);
      v34 = *v33;
      if ( v27 == *v33 )
      {
LABEL_35:
        v35 = v33 + 1;
        v36 = v33[1];
        goto LABEL_36;
      }
      while ( v34 != -4096 )
      {
        if ( !v31 && v34 == -8192 )
          v31 = v33;
        v32 = (v26 - 1) & (v30 + v32);
        v33 = (__int64 *)(v29 + 24LL * v32);
        v34 = *v33;
        if ( v27 == *v33 )
          goto LABEL_35;
        ++v30;
      }
      if ( v31 )
        v33 = v31;
      ++*(_QWORD *)(a1 + 872);
      v38 = *(_DWORD *)(a1 + 888) + 1;
      if ( 4 * v38 < 3 * v26 )
      {
        if ( v26 - *(_DWORD *)(a1 + 892) - v38 > v26 >> 3 )
        {
LABEL_55:
          *(_DWORD *)(a1 + 888) = v38;
          if ( *v33 != -4096 )
            --*(_DWORD *)(a1 + 892);
          *v33 = v27;
          v35 = v33 + 1;
          v33[1] = 0;
          *((_DWORD *)v33 + 4) = 0;
          v36 = 0;
LABEL_36:
          if ( a4 == v36 )
          {
            ++*((_DWORD *)v35 + 2);
            v25 = 0;
          }
          else
          {
            *((_DWORD *)v35 + 2) = 1;
            v25 = 0;
            *v35 = a4;
          }
          goto LABEL_39;
        }
        sub_32B0430(v28, v26);
        v39 = *(_DWORD *)(a1 + 896);
        if ( v39 )
        {
          v40 = v39 - 1;
          v41 = *(_QWORD *)(a1 + 880);
          v42 = 1;
          v43 = v40 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v38 = *(_DWORD *)(a1 + 888) + 1;
          v44 = 0;
          v33 = (__int64 *)(v41 + 24LL * v43);
          v45 = *v33;
          if ( v27 != *v33 )
          {
            while ( v45 != -4096 )
            {
              if ( !v44 && v45 == -8192 )
                v44 = v33;
              v43 = v40 & (v42 + v43);
              v33 = (__int64 *)(v41 + 24LL * v43);
              v45 = *v33;
              if ( v27 == *v33 )
                goto LABEL_55;
              ++v42;
            }
            if ( v44 )
              v33 = v44;
          }
          goto LABEL_55;
        }
LABEL_82:
        ++*(_DWORD *)(a1 + 888);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 872);
    }
    sub_32B0430(v28, 2 * v26);
    v46 = *(_DWORD *)(a1 + 896);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 880);
      v49 = v47 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v38 = *(_DWORD *)(a1 + 888) + 1;
      v33 = (__int64 *)(v48 + 24LL * v49);
      v50 = *v33;
      if ( v27 != *v33 )
      {
        v51 = 1;
        v52 = 0;
        while ( v50 != -4096 )
        {
          if ( v50 == -8192 && !v52 )
            v52 = v33;
          v49 = v47 & (v51 + v49);
          v33 = (__int64 *)(v48 + 24LL * v49);
          v50 = *v33;
          if ( v27 == *v33 )
            goto LABEL_55;
          ++v51;
        }
        if ( v52 )
          v33 = v52;
      }
      goto LABEL_55;
    }
    goto LABEL_82;
  }
LABEL_39:
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( !v65 )
    _libc_free((unsigned __int64)v62);
  return v25;
}
