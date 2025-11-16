// Function: sub_8B7F20
// Address: 0x8b7f20
//
__int64 *__fastcall sub_8B7F20(
        unsigned __int64 a1,
        __m128i *a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        int a7,
        _DWORD *a8)
{
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 *v16; // [rsp+0h] [rbp-20h] BYREF
  __m128i *v17; // [rsp+8h] [rbp-18h] BYREF

  for ( ; a2[8].m128i_i8[12] == 12; a2 = (__m128i *)a2[10].m128i_i64[0] )
    ;
  v11 = *(_BYTE *)(a1 + 80);
  if ( v11 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v11 = *(_BYTE *)(a1 + 80);
  }
  if ( v11 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v11 = *(_BYTE *)(a1 + 80);
  }
  switch ( v11 )
  {
    case 4:
    case 5:
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v12 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v13 = **(_QWORD **)(v12 + 328);
  *a8 = 0;
  if ( (unsigned int)sub_8B7B10((__m128i *)a1, a2, (__int64 *)&v17, (__int64 *)&v16, v13, a3, a5, a6) )
  {
    v14 = v16;
    if ( !v16 )
    {
      v16 = sub_8B6180(a1, v17, a7);
      sub_894C00((__int64)v16);
      *a8 = 1;
      v14 = v16;
    }
    sub_88FC00(v14[11], (__int64 **)v17, a4);
  }
  return v16;
}
