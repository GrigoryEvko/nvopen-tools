// Function: sub_1CCFA00
// Address: 0x1ccfa00
//
__int64 __fastcall sub_1CCFA00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r10
  _QWORD *v17; // rdi
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rax
  int v21; // r8d
  int v22; // r9d
  __int64 result; // rax
  const void *i; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  int v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v13 = *(_QWORD *)(a2 + 8);
  v25 = a1 + 160;
  for ( i = (const void *)(a1 + 176); v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    while ( 1 )
    {
      v14 = sub_1648700(v13);
      v15 = (__int64)v14;
      if ( *((_BYTE *)v14 + 16) == 26 && (*((_DWORD *)v14 + 5) & 0xFFFFFFF) != 1 )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
        goto LABEL_14;
    }
    if ( *(_DWORD *)(a3 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(a3 + 24) )
      {
LABEL_18:
        v16 = *(_QWORD *)(v15 - 48);
        goto LABEL_9;
      }
    }
    else
    {
      v26 = *(_DWORD *)(a3 + 32);
      if ( v26 == (unsigned int)sub_16A57B0(a3 + 24) )
        goto LABEL_18;
    }
    v16 = *(_QWORD *)(v15 - 24);
LABEL_9:
    v27 = v16;
    v17 = sub_1648A60(56, 1u);
    if ( v17 )
      sub_15F8320((__int64)v17, v27, v15);
    v20 = *(unsigned int *)(a1 + 168);
    if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 172) )
    {
      sub_16CD150(v25, i, 0, 8, v18, v19);
      v20 = *(unsigned int *)(a1 + 168);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v20) = v15;
    ++*(_DWORD *)(a1 + 168);
  }
LABEL_14:
  sub_164D160(a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  result = *(unsigned int *)(a1 + 168);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 172) )
  {
    sub_16CD150(v25, (const void *)(a1 + 176), 0, 8, v21, v22);
    result = *(unsigned int *)(a1 + 168);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 168);
  return result;
}
