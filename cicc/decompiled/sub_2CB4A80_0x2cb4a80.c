// Function: sub_2CB4A80
// Address: 0x2cb4a80
//
__int64 __fastcall sub_2CB4A80(__int64 a1, _BYTE *a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  *a4 = 0;
  *a3 = 0;
  result = (unsigned __int8)*a2;
  if ( (unsigned __int8)result > 0x1Cu || (_BYTE)result == 22 )
  {
    result = (__int64)sub_2CB1240(a1, a2, a4);
    *a3 = result;
    if ( *(_BYTE *)result == 61 )
    {
      v9[0] = 0;
      *a3 = sub_2CB3CF0(*(_QWORD *)(result - 32), v9, a1, a5);
      result = v9[0];
      *a4 += v9[0];
    }
  }
  return result;
}
