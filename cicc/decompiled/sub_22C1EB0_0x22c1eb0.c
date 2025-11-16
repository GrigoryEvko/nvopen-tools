// Function: sub_22C1EB0
// Address: 0x22c1eb0
//
__int64 __fastcall sub_22C1EB0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rdx
  bool v13; // zf
  char v14; // dl
  __int64 v15; // rbx
  __int64 i; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // r15
  __int64 v23; // r8
  unsigned __int64 *v24; // rdi
  int v26; // r10d
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  _QWORD v31[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v32; // [rsp+28h] [rbp-78h]
  char v33; // [rsp+30h] [rbp-70h]
  void *v34; // [rsp+40h] [rbp-60h]
  __int64 v35; // [rsp+48h] [rbp-58h] BYREF
  __int64 v36; // [rsp+50h] [rbp-50h]
  __int64 v37; // [rsp+58h] [rbp-48h]
  char v38; // [rsp+60h] [rbp-40h]

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
  if ( !v5 )
    return sub_22BDEB0(a1);
  v8 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v35 = 2;
  v29 = 48 * v4;
  v9 = v5 + 48 * v4;
  v36 = 0;
  v10 = v7 + 48 * v8;
  v34 = &unk_49DE8C0;
  v38 = 0;
  v37 = -4096;
  if ( v7 != v10 )
  {
    do
    {
      if ( v7 )
      {
        v11 = v35;
        *(_QWORD *)(v7 + 16) = 0;
        *(_QWORD *)(v7 + 8) = v11 & 6;
        v12 = v37;
        v13 = v37 == -4096;
        *(_QWORD *)(v7 + 24) = v37;
        if ( v12 != 0 && !v13 && v12 != -8192 )
        {
          v30 = v7;
          sub_BD6050((unsigned __int64 *)(v7 + 8), v11 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v30;
        }
        v14 = v38;
        *(_QWORD *)v7 = &unk_49DE8C0;
        *(_BYTE *)(v7 + 32) = v14;
      }
      v7 += 48;
    }
    while ( v10 != v7 );
    if ( !v38 )
    {
      v34 = &unk_49DB368;
      if ( v37 != -8192 && v37 != -4096 )
      {
        if ( v37 )
          sub_BD60C0(&v35);
      }
    }
  }
  v31[1] = 0;
  v31[0] = 2;
  v33 = 0;
  v32 = -4096;
  v35 = 2;
  v36 = 0;
  v34 = &unk_49DE8C0;
  v38 = 0;
  v37 = -8192;
  if ( v9 == v5 )
    goto LABEL_32;
  v15 = v5;
  for ( i = -4096; ; i = v32 )
  {
    v18 = *(_QWORD *)(v15 + 24);
    if ( v18 != i && v18 != v37 )
    {
      v19 = *(_DWORD *)(a1 + 24);
      if ( !v19 )
        BUG();
      v20 = *(_QWORD *)(a1 + 8);
      v21 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = v20 + 48LL * v21;
      v23 = *(_QWORD *)(v22 + 24);
      if ( v23 == v18 )
      {
LABEL_23:
        v24 = (unsigned __int64 *)(v22 + 8);
        if ( !*(_BYTE *)(v22 + 32) )
          goto LABEL_28;
      }
      else
      {
        v26 = 1;
        v27 = 0;
        while ( v23 != -4096 )
        {
          if ( !v27 && v23 == -8192 )
            v27 = v22;
          v21 = (v19 - 1) & (v26 + v21);
          v22 = v20 + 48LL * v21;
          v23 = *(_QWORD *)(v22 + 24);
          if ( v18 == v23 )
            goto LABEL_23;
          ++v26;
        }
        if ( v27 )
          v22 = v27;
        v24 = (unsigned __int64 *)(v22 + 8);
        if ( !*(_BYTE *)(v22 + 32) )
        {
          v28 = *(_QWORD *)(v22 + 24);
          if ( v18 != v28 )
          {
            if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
            {
              sub_BD60C0(v24);
              v18 = *(_QWORD *)(v15 + 24);
              v24 = (unsigned __int64 *)(v22 + 8);
            }
            goto LABEL_25;
          }
LABEL_28:
          *(_BYTE *)(v22 + 32) = *(_BYTE *)(v15 + 32);
          *(_QWORD *)(v22 + 40) = *(_QWORD *)(v15 + 40);
          *(_QWORD *)(v15 + 40) = 0;
          ++*(_DWORD *)(a1 + 16);
          sub_22C1BD0((unsigned __int64 *)(v15 + 40));
          goto LABEL_29;
        }
      }
      *(_QWORD *)(v22 + 24) = 0;
      v18 = *(_QWORD *)(v15 + 24);
      if ( v18 )
      {
LABEL_25:
        *(_QWORD *)(v22 + 24) = v18;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          sub_BD6050(v24, *(_QWORD *)(v15 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_28;
      }
      goto LABEL_28;
    }
LABEL_29:
    if ( !*(_BYTE *)(v15 + 32) )
      break;
    *(_QWORD *)(v15 + 24) = 0;
    v15 += 48;
    *(_QWORD *)(v15 - 48) = &unk_49DB368;
    if ( v9 == v15 )
      goto LABEL_31;
LABEL_18:
    ;
  }
  v17 = *(_QWORD *)(v15 + 24);
  *(_QWORD *)v15 = &unk_49DB368;
  if ( v17 != 0 && v17 != -8192 && v17 != -4096 )
    sub_BD60C0((_QWORD *)(v15 + 8));
  v15 += 48;
  if ( v9 != v15 )
    goto LABEL_18;
LABEL_31:
  if ( !v38 )
  {
    v34 = &unk_49DB368;
    if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
    {
      sub_BD60C0(&v35);
      if ( !v33 )
        goto LABEL_33;
      return sub_C7D6A0(v5, v29, 8);
    }
  }
LABEL_32:
  if ( !v33 )
  {
LABEL_33:
    if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
      sub_BD60C0(v31);
  }
  return sub_C7D6A0(v5, v29, 8);
}
