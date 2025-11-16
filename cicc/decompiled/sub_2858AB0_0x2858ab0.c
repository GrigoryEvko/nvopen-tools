// Function: sub_2858AB0
// Address: 0x2858ab0
//
char __fastcall sub_2858AB0(__int64 a1, char *a2, __int64 a3, __int64 *a4, unsigned int a5)
{
  __int16 v7; // ax
  __int64 v8; // rdi
  unsigned int v9; // ebx
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r15
  char v17; // bl
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v21; // rdi
  int v22; // eax
  unsigned __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  unsigned int v27; // eax
  unsigned __int64 *v28; // rdx
  unsigned __int64 *v29; // [rsp+8h] [rbp-38h]

  v7 = *(_WORD *)(a1 + 24);
  if ( v7 )
  {
    if ( v7 != 6 )
      return 0;
    if ( *(_QWORD *)(a1 + 40) != 2 )
      return 0;
    v24 = *(__int64 **)(a1 + 32);
    if ( *(_WORD *)(v24[1] + 24) != 1 )
      return 0;
    v25 = *v24;
    if ( *(_WORD *)(v25 + 24) )
      return 0;
    v26 = *(_QWORD *)(v25 + 32);
    if ( (unsigned int)sub_BCB060(*(_QWORD *)(v26 + 8)) > 0x40 )
      return 0;
    v27 = *(_DWORD *)(v26 + 32);
    v28 = *(unsigned __int64 **)(v26 + 24);
    if ( v27 > 0x40 )
    {
      v16 = *v28;
    }
    else
    {
      v16 = 0;
      if ( v27 )
      {
        v11 = 64 - v27;
        v16 = (__int64)((_QWORD)v28 << (64 - (unsigned __int8)v27)) >> (64 - (unsigned __int8)v27);
      }
    }
    v17 = 1;
    goto LABEL_10;
  }
  v8 = *(_QWORD *)(a1 + 32);
  v9 = *(_DWORD *)(v8 + 32);
  v10 = *(_QWORD *)(v8 + 24);
  v11 = v9 - 1;
  v12 = 1LL << ((unsigned __int8)v9 - 1);
  if ( v9 > 0x40 )
  {
    v21 = v8 + 24;
    v29 = (unsigned __int64 *)v10;
    if ( (*(_QWORD *)(v10 + 8LL * ((unsigned int)v11 >> 6)) & v12) != 0 )
      v22 = sub_C44500(v21);
    else
      v22 = sub_C444A0(v21);
    if ( v9 + 1 - v22 > 0x40 )
      return 0;
    v16 = *v29;
    goto LABEL_9;
  }
  if ( (v12 & v10) != 0 )
  {
    if ( !v9 )
    {
      v16 = 0;
      goto LABEL_9;
    }
    v13 = 64;
    v11 = 64 - v9;
    v14 = v10 << (64 - (unsigned __int8)v9);
    if ( v14 != -1 )
    {
      _BitScanReverse64(&v15, ~v14);
      v13 = v15 ^ 0x3F;
    }
    if ( v9 + 1 - v13 > 0x40 )
      return 0;
  }
  else
  {
    if ( v10 )
    {
      _BitScanReverse64(&v23, v10);
      if ( (unsigned int)v23 == 0x3F )
        return 0;
    }
    v16 = 0;
    if ( !v9 )
      goto LABEL_9;
    v11 = 64 - v9;
    v14 = v10 << (64 - (unsigned __int8)v9);
  }
  v16 = v14 >> v11;
LABEL_9:
  v17 = 0;
LABEL_10:
  if ( (unsigned __int8)sub_2851CA0((__int64)a4, a2, a3, v11, a5) )
  {
    v18 = sub_284F800((__int64)a4, (__int64)a2, a3);
    return sub_2850840(a4, 2u, v18, v19, v16, v17, 0);
  }
  return 0;
}
