// Function: sub_395D8D0
// Address: 0x395d8d0
//
__int64 __fastcall sub_395D8D0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  *a4 = 0;
  *a3 = 0;
  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result > 0x17u || (_BYTE)result == 17 )
  {
    result = sub_395AC60(a1, a2, a4);
    *a3 = result;
    if ( *(_BYTE *)(result + 16) == 54 )
    {
      v9[0] = 0;
      *a3 = sub_395CFD0(*(unsigned __int16 **)(result - 24), v9, a1, a5);
      result = v9[0];
      *a4 += v9[0];
    }
  }
  return result;
}
