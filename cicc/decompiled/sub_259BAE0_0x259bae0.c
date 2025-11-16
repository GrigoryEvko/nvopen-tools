// Function: sub_259BAE0
// Address: 0x259bae0
//
__int64 __fastcall sub_259BAE0(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v2; // r13d
  char v4; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v5[3]; // [rsp+10h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_B46420(a2) )
    return 1;
  if ( (unsigned __int8)sub_B46490(a2) )
    return 1;
  sub_250D230((unsigned __int64 *)v5, a2, 5, 0);
  v2 = sub_259B8C0(*a1, a1[1], v5, 1, &v4, 0, 0);
  if ( (_BYTE)v2 )
    return 1;
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 6) )
    return v2;
  return (unsigned int)sub_B49560(a2, 6) ^ 1;
}
