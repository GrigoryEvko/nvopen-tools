// Function: sub_2516380
// Address: 0x2516380
//
__int64 __fastcall sub_2516380(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  char v11; // [rsp+Ch] [rbp-34h] BYREF
  _QWORD v12[6]; // [rsp+10h] [rbp-30h] BYREF

  v8 = *a2;
  v11 = a5;
  v9 = v8 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v8 & 3) == 3 )
    v9 = *(_QWORD *)(v9 + 24);
  v12[0] = sub_BD5C60(v9);
  v12[1] = &v11;
  return sub_2515E30(
           a1,
           a2,
           a3,
           a4,
           (unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD *, __int64 **))sub_2506280,
           (__int64)v12);
}
