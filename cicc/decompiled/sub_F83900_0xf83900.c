// Function: sub_F83900
// Address: 0xf83900
//
__int64 __fastcall sub_F83900(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r12
  void *v10; // r8
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax
  bool v15; // zf
  unsigned __int8 v16; // al
  __int64 v17; // rbx
  __int64 i; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // ecx
  int v22; // esi
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 v25; // r15
  __int64 v26; // r8
  unsigned __int64 *v27; // rdi
  __int64 result; // rax
  __int64 v29; // r12
  __int64 v30; // rsi
  __int64 v31; // rax
  int v32; // r10d
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // [rsp+10h] [rbp-A0h]
  void *v36; // [rsp+18h] [rbp-98h]
  _QWORD v37[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v38; // [rsp+38h] [rbp-78h]
  char v39; // [rsp+40h] [rbp-70h]
  void *v40; // [rsp+50h] [rbp-60h]
  __int64 v41; // [rsp+58h] [rbp-58h] BYREF
  __int64 v42; // [rsp+60h] [rbp-50h]
  __int64 v43; // [rsp+68h] [rbp-48h]
  unsigned __int8 v44; // [rsp+70h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  v8 = v7;
  if ( !v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v41 = 2;
    result = *(unsigned int *)(a1 + 24);
    v42 = 0;
    v40 = &unk_49E51C0;
    v44 = 0;
    v43 = -4096;
    v29 = v8 + 48 * result;
    if ( v8 != v29 )
    {
      do
      {
        if ( v8 )
        {
          v30 = v41;
          *(_QWORD *)(v8 + 16) = 0;
          *(_QWORD *)(v8 + 8) = v30 & 6;
          v31 = v43;
          v15 = v43 == -4096;
          *(_QWORD *)(v8 + 24) = v43;
          if ( v31 != 0 && !v15 && v31 != -8192 )
            sub_BD6050((unsigned __int64 *)(v8 + 8), v30 & 0xFFFFFFFFFFFFFFF8LL);
          result = v44;
          *(_QWORD *)v8 = &unk_49E51C0;
          *(_BYTE *)(v8 + 32) = result;
        }
        v8 += 48;
      }
      while ( v29 != v8 );
      if ( !v44 )
      {
        result = v43;
        v40 = &unk_49DB368;
        if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
          return sub_BD60C0(&v41);
      }
    }
    return result;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v41 = 2;
  v35 = 48 * v4;
  v9 = v5 + 48 * v4;
  v42 = 0;
  v10 = &unk_49E51C0;
  v11 = *(unsigned int *)(a1 + 24);
  v40 = &unk_49E51C0;
  v44 = 0;
  v43 = -4096;
  v12 = v8 + 48 * v11;
  if ( v8 != v12 )
  {
    do
    {
      if ( v8 )
      {
        v13 = v41;
        *(_QWORD *)(v8 + 16) = 0;
        *(_QWORD *)(v8 + 8) = v13 & 6;
        v14 = v43;
        v15 = v43 == -4096;
        *(_QWORD *)(v8 + 24) = v43;
        if ( v14 != 0 && !v15 && v14 != -8192 )
        {
          v36 = v10;
          sub_BD6050((unsigned __int64 *)(v8 + 8), v13 & 0xFFFFFFFFFFFFFFF8LL);
          v10 = v36;
        }
        v16 = v44;
        *(_QWORD *)v8 = v10;
        *(_BYTE *)(v8 + 32) = v16;
      }
      v8 += 48;
    }
    while ( v12 != v8 );
    if ( !v44 )
    {
      v40 = &unk_49DB368;
      if ( v43 != -4096 && v43 != 0 && v43 != -8192 )
        sub_BD60C0(&v41);
    }
  }
  v37[1] = 0;
  v37[0] = 2;
  v39 = 0;
  v38 = -4096;
  v41 = 2;
  v42 = 0;
  v40 = &unk_49E51C0;
  v44 = 0;
  v43 = -8192;
  if ( v9 == v5 )
    goto LABEL_32;
  v17 = v5;
  for ( i = -4096; ; i = v38 )
  {
    v20 = *(_QWORD *)(v17 + 24);
    if ( i != v20 && v20 != v43 )
    {
      v21 = *(_DWORD *)(a1 + 24);
      if ( !v21 )
        BUG();
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v24 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v25 = v23 + 48LL * v24;
      v26 = *(_QWORD *)(v25 + 24);
      if ( v20 == v26 )
      {
LABEL_23:
        v27 = (unsigned __int64 *)(v25 + 8);
        if ( !*(_BYTE *)(v25 + 32) )
          goto LABEL_28;
      }
      else
      {
        v32 = 1;
        v33 = 0;
        while ( v26 != -4096 )
        {
          if ( !v33 && v26 == -8192 )
            v33 = v25;
          v24 = v22 & (v32 + v24);
          v25 = v23 + 48LL * v24;
          v26 = *(_QWORD *)(v25 + 24);
          if ( v20 == v26 )
            goto LABEL_23;
          ++v32;
        }
        if ( v33 )
          v25 = v33;
        v27 = (unsigned __int64 *)(v25 + 8);
        if ( !*(_BYTE *)(v25 + 32) )
        {
          v34 = *(_QWORD *)(v25 + 24);
          if ( v20 != v34 )
          {
            if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
            {
              sub_BD60C0(v27);
              v20 = *(_QWORD *)(v17 + 24);
              v27 = (unsigned __int64 *)(v25 + 8);
            }
            goto LABEL_25;
          }
LABEL_28:
          *(_BYTE *)(v25 + 32) = *(_BYTE *)(v17 + 32);
          *(_QWORD *)(v25 + 40) = *(_QWORD *)(v17 + 40);
          ++*(_DWORD *)(a1 + 16);
          goto LABEL_29;
        }
      }
      *(_QWORD *)(v25 + 24) = 0;
      v20 = *(_QWORD *)(v17 + 24);
      if ( v20 )
      {
LABEL_25:
        *(_QWORD *)(v25 + 24) = v20;
        if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
          sub_BD6050(v27, *(_QWORD *)(v17 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_28;
      }
      goto LABEL_28;
    }
LABEL_29:
    if ( !*(_BYTE *)(v17 + 32) )
      break;
    *(_QWORD *)(v17 + 24) = 0;
    v17 += 48;
    *(_QWORD *)(v17 - 48) = &unk_49DB368;
    if ( v9 == v17 )
      goto LABEL_31;
LABEL_18:
    ;
  }
  v19 = *(_QWORD *)(v17 + 24);
  *(_QWORD *)v17 = &unk_49DB368;
  if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
    sub_BD60C0((_QWORD *)(v17 + 8));
  v17 += 48;
  if ( v9 != v17 )
    goto LABEL_18;
LABEL_31:
  if ( !v44 )
  {
    v40 = &unk_49DB368;
    if ( v43 != -4096 && v43 != 0 && v43 != -8192 )
    {
      sub_BD60C0(&v41);
      if ( !v39 )
        goto LABEL_33;
      return sub_C7D6A0(v5, v35, 8);
    }
  }
LABEL_32:
  if ( !v39 )
  {
LABEL_33:
    if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
      sub_BD60C0(v37);
  }
  return sub_C7D6A0(v5, v35, 8);
}
