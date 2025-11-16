// Function: sub_8B8060
// Address: 0x8b8060
//
__int64 __fastcall sub_8B8060(__int64 a1, __m128i *a2, __int64 a3, int a4, int a5)
{
  char v5; // al
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 v9; // [rsp+0h] [rbp-20h] BYREF
  __int64 *i; // [rsp+8h] [rbp-18h] BYREF

  for ( i = 0; a2[8].m128i_i8[12] == 12; a2 = (__m128i *)a2[10].m128i_i64[0] )
    ;
  v5 = *(_BYTE *)(a1 + 80);
  if ( v5 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v5 = *(_BYTE *)(a1 + 80);
  }
  if ( v5 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v5 = *(_BYTE *)(a1 + 80);
  }
  switch ( v5 )
  {
    case 4:
    case 5:
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v6 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v7 = sub_8B7B10((__m128i *)a1, a2, (__int64 *)&i, &v9, **(_QWORD **)(v6 + 328), a3, a4, a5);
  if ( i )
    sub_725130(i);
  return v7;
}
