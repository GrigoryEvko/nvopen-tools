// Function: sub_2C489D0
// Address: 0x2c489d0
//
__int64 __fastcall sub_2C489D0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 *v2; // r12
  __int64 v3; // r14
  __int64 v4; // rbx
  char v5; // dl
  int v6; // esi
  __int64 *v7; // rcx
  unsigned int v8; // eax
  __int64 *v9; // r9
  __int64 v10; // r8
  unsigned int v11; // r12d
  unsigned int v13; // esi
  unsigned int v14; // eax
  __int64 *v15; // rdi
  unsigned int v16; // ecx
  int v17; // r10d
  __int64 *v18; // rsi
  int v19; // ecx
  unsigned int v20; // edx
  __int64 v21; // r8
  int v22; // r9d
  __int64 *v23; // rax
  __int64 *v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // r8
  int v28; // r9d
  __int64 v29; // [rsp+0h] [rbp-70h] BYREF
  __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 *v31; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v32; // [rsp+18h] [rbp-58h]
  char v33; // [rsp+50h] [rbp-20h] BYREF

  v1 = (__int64 *)&v31;
  v29 = 0;
  v30 = 1;
  do
    *v1++ = -4096;
  while ( v1 != (__int64 *)&v33 );
  v2 = *(__int64 **)a1;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( v3 != *(_QWORD *)a1 )
  {
    do
    {
      v4 = *v2;
      v5 = v30 & 1;
      if ( (v30 & 1) != 0 )
      {
        v6 = 7;
        v7 = (__int64 *)&v31;
      }
      else
      {
        v13 = v32;
        v7 = v31;
        if ( !v32 )
        {
          v14 = v30;
          ++v29;
          v15 = 0;
          v16 = ((unsigned int)v30 >> 1) + 1;
          goto LABEL_14;
        }
        v6 = v32 - 1;
      }
      v8 = v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v9 = &v7[v8];
      v10 = *v9;
      if ( v4 == *v9 )
      {
LABEL_7:
        v11 = 1;
        goto LABEL_8;
      }
      v17 = 1;
      v15 = 0;
      while ( v10 != -4096 )
      {
        if ( v10 != -8192 || v15 )
          v9 = v15;
        v8 = v6 & (v17 + v8);
        v10 = v7[v8];
        if ( v4 == v10 )
          goto LABEL_7;
        ++v17;
        v15 = v9;
        v9 = &v7[v8];
      }
      v14 = v30;
      if ( !v15 )
        v15 = v9;
      ++v29;
      v16 = ((unsigned int)v30 >> 1) + 1;
      if ( !v5 )
      {
        v13 = v32;
LABEL_14:
        if ( 4 * v16 < 3 * v13 )
          goto LABEL_15;
        goto LABEL_26;
      }
      v13 = 8;
      if ( 4 * v16 < 0x18 )
      {
LABEL_15:
        if ( v13 - HIDWORD(v30) - v16 > v13 >> 3 )
          goto LABEL_16;
        sub_2C485C0((__int64)&v29, v13);
        if ( (v30 & 1) != 0 )
        {
          v25 = 7;
          v24 = (__int64 *)&v31;
        }
        else
        {
          v24 = v31;
          if ( !v32 )
          {
LABEL_59:
            LODWORD(v30) = (2 * ((unsigned int)v30 >> 1) + 2) | v30 & 1;
            BUG();
          }
          v25 = v32 - 1;
        }
        v26 = v25 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v15 = &v24[v26];
        v14 = v30;
        v27 = *v15;
        if ( v4 == *v15 )
          goto LABEL_16;
        v28 = 1;
        v23 = 0;
        while ( v27 != -4096 )
        {
          if ( !v23 && v27 == -8192 )
            v23 = v15;
          v26 = v25 & (v28 + v26);
          v15 = &v24[v26];
          v27 = *v15;
          if ( v4 == *v15 )
            goto LABEL_34;
          ++v28;
        }
        goto LABEL_32;
      }
LABEL_26:
      sub_2C485C0((__int64)&v29, 2 * v13);
      if ( (v30 & 1) != 0 )
      {
        v19 = 7;
        v18 = (__int64 *)&v31;
      }
      else
      {
        v18 = v31;
        if ( !v32 )
          goto LABEL_59;
        v19 = v32 - 1;
      }
      v20 = v19 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = &v18[v20];
      v14 = v30;
      v21 = *v15;
      if ( v4 == *v15 )
        goto LABEL_16;
      v22 = 1;
      v23 = 0;
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v23 )
          v23 = v15;
        v20 = v19 & (v22 + v20);
        v15 = &v18[v20];
        v21 = *v15;
        if ( v4 == *v15 )
          goto LABEL_34;
        ++v22;
      }
LABEL_32:
      if ( v23 )
        v15 = v23;
LABEL_34:
      v14 = v30;
LABEL_16:
      LODWORD(v30) = (2 * (v14 >> 1) + 2) | v14 & 1;
      if ( *v15 != -4096 )
        --HIDWORD(v30);
      ++v2;
      *v15 = v4;
    }
    while ( (__int64 *)v3 != v2 );
  }
  v11 = 0;
  v5 = v30 & 1;
LABEL_8:
  if ( !v5 )
    sub_C7D6A0((__int64)v31, 8LL * v32, 8);
  return v11;
}
