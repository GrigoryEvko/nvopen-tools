// Function: sub_2575960
// Address: 0x2575960
//
__int64 __fastcall sub_2575960(__int64 a1, int a2)
{
  __int64 v3; // r15
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r12
  unsigned int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // r15
  unsigned int i; // edx
  __int64 v15; // rax
  unsigned int v16; // eax
  const void *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // rdi
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  unsigned int v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+28h] [rbp-58h] BYREF
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v26 = v3;
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = v6;
  v7 = v6;
  if ( !v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v19 = *(unsigned int *)(a1 + 24);
    v31 = 0;
    v30 = -1;
    v20 = v6 + 16 * v19;
    if ( v6 == v20 )
      return sub_969240(&v30);
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_38;
      v21 = v31;
      *(_DWORD *)(v7 + 8) = v31;
      if ( v21 <= 0x40 )
      {
        *(_QWORD *)v7 = v30;
LABEL_38:
        v7 += 16;
        if ( v20 == v7 )
          return sub_969240(&v30);
      }
      else
      {
        v22 = v7;
        v7 += 16;
        sub_C43780(v22, (const void **)&v30);
        if ( v20 == v7 )
          return sub_969240(&v30);
      }
    }
  }
  v8 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v23 = 16 * v4;
  v8 *= 16;
  v31 = 0;
  v9 = v6 + v8;
  v30 = -1;
  v10 = 16 * v4 + v26;
  if ( v6 != v6 + v8 )
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_7;
      v11 = v31;
      *(_DWORD *)(v7 + 8) = v31;
      if ( v11 <= 0x40 )
      {
        *(_QWORD *)v7 = v30;
LABEL_7:
        v7 += 16;
        if ( v9 == v7 )
          break;
      }
      else
      {
        v12 = v7;
        v7 += 16;
        sub_C43780(v12, (const void **)&v30);
        if ( v9 == v7 )
          break;
      }
    }
  }
  sub_969240(&v30);
  v29 = 0;
  v28 = -1;
  v31 = 0;
  v30 = -2;
  if ( v10 != v26 )
  {
    v13 = v26;
    for ( i = 0; ; i = v29 )
    {
      v16 = *(_DWORD *)(v13 + 8);
      if ( v16 != i )
        break;
      if ( i <= 0x40 )
      {
        v18 = *(const void **)v13;
        if ( *(_QWORD *)v13 == v28 )
          goto LABEL_18;
        if ( i != v31 )
          goto LABEL_14;
LABEL_30:
        if ( v18 == (const void *)v30 )
          goto LABEL_18;
        goto LABEL_14;
      }
      v25 = i;
      if ( sub_C43C50(v13, (const void **)&v28) )
        goto LABEL_25;
      if ( v25 == v31 )
      {
LABEL_24:
        if ( sub_C43C50(v13, (const void **)&v30) )
          goto LABEL_25;
      }
LABEL_14:
      sub_2567ED0(a1, v13, &v27);
      v15 = v27;
      if ( *(_DWORD *)(v27 + 8) > 0x40u )
      {
        if ( *(_QWORD *)v27 )
        {
          v24 = v27;
          j_j___libc_free_0_0(*(_QWORD *)v27);
          v15 = v24;
        }
      }
      *(_QWORD *)v15 = *(_QWORD *)v13;
      *(_DWORD *)(v15 + 8) = *(_DWORD *)(v13 + 8);
      *(_DWORD *)(v13 + 8) = 0;
      ++*(_DWORD *)(a1 + 16);
      if ( *(_DWORD *)(v13 + 8) <= 0x40u )
      {
LABEL_18:
        v13 += 16;
        if ( v10 == v13 )
          goto LABEL_27;
        continue;
      }
LABEL_25:
      if ( !*(_QWORD *)v13 )
        goto LABEL_18;
      j_j___libc_free_0_0(*(_QWORD *)v13);
      v13 += 16;
      if ( v10 == v13 )
        goto LABEL_27;
    }
    if ( v16 != v31 )
      goto LABEL_14;
    if ( v16 > 0x40 )
      goto LABEL_24;
    v18 = *(const void **)v13;
    goto LABEL_30;
  }
LABEL_27:
  sub_969240(&v30);
  sub_969240(&v28);
  return sub_C7D6A0(v26, v23, 8);
}
