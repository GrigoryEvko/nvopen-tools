// Function: sub_324A0C0
// Address: 0x324a0c0
//
__int64 __fastcall sub_324A0C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v5; // r14d
  __int64 v6[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_3249FA0(a1, a2, 60);
  if ( (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v5 = (unsigned __int16)sub_3220AA0(a1[26]), result = sub_E06A90(105), v5 >= (unsigned int)result) )
  {
    v6[1] = a3;
    v6[0] = 0x20006900000001LL;
    return sub_3248F80((unsigned __int64 **)(a2 + 8), a1 + 11, v6);
  }
  return result;
}
