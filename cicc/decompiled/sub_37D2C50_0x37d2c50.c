// Function: sub_37D2C50
// Address: 0x37d2c50
//
__int64 __fastcall sub_37D2C50(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // bl
  unsigned int v5; // eax
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r15
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rbx
  __int64 result; // rax
  __int64 *v20; // r13
  __int64 v21; // rdi
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 *v24; // rbx
  __int64 *v25; // rax
  __int64 *v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // r15
  __int64 *v32; // rcx
  __int64 v33; // rax
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 *v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v38[14]; // [rsp+30h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
LABEL_21:
    v20 = (__int64 *)(a1 + 16);
    v21 = a1 + 80;
    v22 = unk_5051170;
    v23 = qword_5051168;
    goto LABEL_22;
  }
  v5 = sub_AF1560(a2 - 1);
  v2 = v5;
  if ( v5 > 0x40 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v8 = 16LL * v5;
      goto LABEL_5;
    }
    goto LABEL_21;
  }
  if ( !v4 )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 1024;
    v2 = 64;
LABEL_5:
    v9 = sub_C7D670(v8, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
    v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v11 = v6 + 16 * v7;
    v12 = unk_5051170;
    if ( v10 )
    {
      v13 = *(_QWORD **)(a1 + 16);
      v14 = 2LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v13 = (_QWORD *)(a1 + 16);
      v14 = 8;
    }
    for ( i = &v13[v14]; i != v13; v13 += 2 )
    {
      if ( v13 )
        *v13 = v12;
    }
    v16 = unk_5051170;
    v17 = qword_5051168;
    if ( v11 != v6 )
    {
      v18 = v6;
      do
      {
        if ( v16 != *(_QWORD *)v18 && v17 != *(_QWORD *)v18 )
        {
          v34 = v16;
          v35 = v17;
          sub_37BF550(a1, (__int64 *)v18, v38);
          v16 = v34;
          v17 = v35;
          *(_QWORD *)v38[0] = *(_QWORD *)v18;
          *(_DWORD *)(v38[0] + 8LL) = *(_DWORD *)(v18 + 8);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        }
        v18 += 16;
      }
      while ( v11 != v18 );
    }
    return sub_C7D6A0(v6, 16 * v7, 8);
  }
  v20 = (__int64 *)(a1 + 16);
  v21 = a1 + 80;
  v2 = 64;
  v22 = unk_5051170;
  v23 = qword_5051168;
LABEL_22:
  v24 = v38;
  v25 = v20;
  v26 = v38;
  do
  {
    v27 = *v25;
    if ( *v25 != v23 && v27 != v22 )
    {
      if ( v26 )
        *v26 = v27;
      v26 += 2;
      *((_DWORD *)v26 - 2) = *((_DWORD *)v25 + 2);
    }
    v25 += 2;
  }
  while ( v25 != (__int64 *)v21 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v28 = unk_5051170;
  v29 = 8;
  if ( v10 )
  {
    v20 = *(__int64 **)(a1 + 16);
    v29 = 2LL * *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v20[v29]; (__int64 *)result != v20; v20 += 2 )
  {
    if ( v20 )
      *v20 = v28;
  }
  v30 = unk_5051170;
  v31 = qword_5051168;
  if ( v26 != v38 )
  {
    v32 = &v37;
    do
    {
      result = *v24;
      if ( *v24 != v30 && result != v31 )
      {
        v36 = v32;
        sub_37BF550(a1, v24, v32);
        v32 = v36;
        *(_QWORD *)v37 = *v24;
        *(_DWORD *)(v37 + 8) = *((_DWORD *)v24 + 2);
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
      }
      v24 += 2;
    }
    while ( v26 != v24 );
  }
  return result;
}
