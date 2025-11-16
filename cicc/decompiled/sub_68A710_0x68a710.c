// Function: sub_68A710
// Address: 0x68a710
//
_BOOL8 __fastcall sub_68A710(int a1, _QWORD *a2)
{
  _BOOL8 result; // rax
  _DWORD v3[28]; // [rsp-1F8h] [rbp-1F8h] BYREF
  __int64 v4; // [rsp-188h] [rbp-188h] BYREF

  result = 0;
  if ( a2 )
  {
    if ( !*a2 )
    {
      sub_82D850(v3);
      sub_839D30((_DWORD)a2, a1, 1, 0, 0, 16778264, 1, 0, 0, (__int64)&v4, 0, (__int64)v3);
      return v3[2] != 7;
    }
  }
  return result;
}
