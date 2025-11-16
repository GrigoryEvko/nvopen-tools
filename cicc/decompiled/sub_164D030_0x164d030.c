// Function: sub_164D030
// Address: 0x164d030
//
void __fastcall sub_164D030(
        __int64 a1,
        __int64 a2,
        char a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *i; // rbx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rcx
  _QWORD *v18; // rdi

  if ( (*(_BYTE *)(a1 + 17) & 1) != 0 )
    sub_164CC90(a1, a2);
  if ( !a3 && (*(_BYTE *)(a1 + 23) & 0x10) != 0 )
    sub_16303F0(a1, a2, a4, a5, a6, a7, a8, a9, a10, a11);
  for ( i = *(__int64 **)(a1 + 8); i; i = *(__int64 **)(a1 + 8) )
  {
    while ( 1 )
    {
      v18 = sub_1648700((__int64)i);
      if ( (unsigned __int8)(*((_BYTE *)v18 + 16) - 4) > 0xCu )
        break;
      sub_15A5060((__int64)v18, (_BYTE *)a1, a2, v17, *(double *)a4.m128_u64, a5, a6);
      i = *(__int64 **)(a1 + 8);
      if ( !i )
        goto LABEL_18;
    }
    if ( *i )
    {
      v14 = i[1];
      v15 = i[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v15 = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
    }
    *i = a2;
    if ( a2 )
    {
      v16 = *(_QWORD *)(a2 + 8);
      i[1] = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = (unsigned __int64)(i + 1) | *(_QWORD *)(v16 + 16) & 3LL;
      i[2] = (a2 + 8) | i[2] & 3;
      *(_QWORD *)(a2 + 8) = i;
    }
  }
LABEL_18:
  if ( *(_BYTE *)(a1 + 16) == 18 )
    sub_157F670(a1, a2);
}
