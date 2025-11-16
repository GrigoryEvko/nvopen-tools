// Function: sub_3531F50
// Address: 0x3531f50
//
__int64 __fastcall sub_3531F50(__m128i *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v7)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  void (__fastcall *v8)(__int64 *, __int64 *, __int64 *, _BYTE *); // [rsp+18h] [rbp-28h]

  sub_A558A0((__int64)a1, *(_QWORD *)(*(_QWORD *)a3 + 40LL), a4);
  v6[0] = a1;
  a1[7].m128i_i64[1] = a2;
  a1[8].m128i_i64[0] = 0;
  a1->m128i_i64[0] = (__int64)&unk_4A38F78;
  a1[7].m128i_i64[0] = *(_QWORD *)a3;
  v8 = sub_3531E60;
  v7 = sub_3531BF0;
  sub_A558F0(a1, (__int64)v6);
  if ( v7 )
    v7(v6, v6, 3);
  v6[0] = a1;
  v8 = sub_3531F10;
  v7 = sub_3531C20;
  sub_A55980(a1, (__int64)v6);
  result = (__int64)v7;
  if ( v7 )
    return v7(v6, v6, 3);
  return result;
}
