// Function: sub_21BCFC0
// Address: 0x21bcfc0
//
__int64 __fastcall sub_21BCFC0(
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
  __int64 *v13; // rbx
  __int64 v14; // r14
  __int64 v15; // r10
  _QWORD *v16; // rdi
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  int v20; // r8d
  int v21; // r9d
  __int64 result; // rax
  const void *i; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  int v25; // [rsp+18h] [rbp-38h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v13 = *(__int64 **)(a2 + 8);
  v24 = a1 + 160;
  for ( i = (const void *)(a1 + 176); v13; v13 = (__int64 *)v13[1] )
  {
    while ( 1 )
    {
      v14 = *v13;
      if ( *(_BYTE *)(*v13 + 16) == 26 && (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != 1 )
        break;
      v13 = (__int64 *)v13[1];
      if ( !v13 )
        goto LABEL_14;
    }
    if ( *(_DWORD *)(a3 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(a3 + 24) )
      {
LABEL_18:
        v15 = *(_QWORD *)(v14 - 48);
        goto LABEL_9;
      }
    }
    else
    {
      v25 = *(_DWORD *)(a3 + 32);
      if ( v25 == (unsigned int)sub_16A57B0(a3 + 24) )
        goto LABEL_18;
    }
    v15 = *(_QWORD *)(v14 - 24);
LABEL_9:
    v26 = v15;
    v16 = sub_1648A60(56, 1u);
    if ( v16 )
      sub_15F8320((__int64)v16, v26, v14);
    v19 = *(unsigned int *)(a1 + 168);
    if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 172) )
    {
      sub_16CD150(v24, i, 0, 8, v17, v18);
      v19 = *(unsigned int *)(a1 + 168);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v19) = v14;
    ++*(_DWORD *)(a1 + 168);
  }
LABEL_14:
  sub_164D160(a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  result = *(unsigned int *)(a1 + 168);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 172) )
  {
    sub_16CD150(v24, (const void *)(a1 + 176), 0, 8, v20, v21);
    result = *(unsigned int *)(a1 + 168);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 168);
  return result;
}
