// Function: sub_2398050
// Address: 0x2398050
//
_QWORD *__fastcall sub_2398050(_QWORD *a1, __int64 a2)
{
  __int64 v3; // [rsp+8h] [rbp-68h] BYREF
  __int64 v4[12]; // [rsp+10h] [rbp-60h] BYREF

  sub_2DD03B0(v4, a2 + 8);
  sub_2397CE0((__int64)&v3, v4);
  *a1 = v3;
  sub_2DD06B0(v4);
  return a1;
}
