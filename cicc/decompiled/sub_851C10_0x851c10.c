// Function: sub_851C10
// Address: 0x851c10
//
__int64 __fastcall sub_851C10(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = sub_822B10(56, a2, a3, a4, a5, a6);
  *(_QWORD *)result = 0;
  *(_DWORD *)(result + 8) = a1;
  if ( a1 == 1 )
  {
    *(_DWORD *)(result + 12) = 0;
    *(_BYTE *)(result + 16) = 0;
  }
  else
  {
    if ( a1 != 2 )
      sub_721090();
    *(_DWORD *)(result + 12) = 22;
  }
  *(_QWORD *)(result + 24) = 0;
  *(_BYTE *)(result + 48) = 0;
  *(_QWORD *)(result + 32) = *(_QWORD *)&dword_4F077C8;
  return result;
}
