// Function: sub_259B770
// Address: 0x259b770
//
__int64 __fastcall sub_259B770(__int64 *a1, unsigned __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // [rsp+Eh] [rbp-22h] BYREF
  char v4; // [rsp+Fh] [rbp-21h] BYREF
  __m128i v5[2]; // [rsp+10h] [rbp-20h] BYREF

  sub_250D230((unsigned __int64 *)v5, a2, 5, 0);
  result = sub_259AB90(*a1, a1[1], v5, 0, &v3, 0, 0);
  if ( (_BYTE)result )
  {
    result = v3;
    if ( !v3 )
      return sub_259B4A0(*a1, a1[1], v5, 0, &v4, 0, 0);
  }
  return result;
}
