// Function: sub_636FC0
// Address: 0x636fc0
//
__int64 __fastcall sub_636FC0(__int64 *a1, __int64 a2, __m128i *a3, __int64 *a4)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rax
  __int64 *v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // [rsp+10h] [rbp-40h] BYREF
  __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a2;
  v16 = (__int64 *)*a1;
  v8 = (__int64)(v16 + 5);
  if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
  {
    *a4 = 0;
  }
  else
  {
    v12 = sub_724D50(10);
    v13 = v16;
    *a4 = v12;
    *(_QWORD *)(v12 + 128) = a2;
    v14 = (_QWORD *)sub_6E1A20(v13);
    v15 = (__int64)v16;
    *(_QWORD *)(*a4 + 64) = *v14;
    if ( *(_BYTE *)(v15 + 8) != 2 )
      *(_QWORD *)(*a4 + 112) = *(_QWORD *)sub_6E1A60(v15);
    *(_BYTE *)(*a4 + 169) = ~a3[2].m128i_i8[11] & 0x20 | *(_BYTE *)(*a4 + 169) & 0xDF;
  }
  while ( *(_BYTE *)(v7 + 140) == 12 )
    v7 = *(_QWORD *)(v7 + 160);
  v9 = sub_72C610(*(unsigned __int8 *)(v7 + 160));
  v16 = (__int64 *)v16[3];
  sub_634B10((__int64 *)&v16, v9, 0, a3, v8, v17);
  if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
    sub_72A690(v17[0], *a4, 0, 0);
  sub_634B10((__int64 *)&v16, v9, 0, a3, v8, v17);
  if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
    sub_72A690(v17[0], *a4, 0, 0);
  if ( v16 )
  {
    v10 = sub_6E1A20(v16);
    sub_6851C0(146, v10);
  }
  result = *(_QWORD *)*a1;
  if ( result && *(_BYTE *)(result + 8) == 3 )
    result = sub_6BBB10(*a1);
  *a1 = result;
  return result;
}
