// Function: sub_2E867B0
// Address: 0x2e867b0
//
unsigned __int64 __fastcall sub_2E867B0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10)
{
  _BOOL4 v10; // r15d
  unsigned __int64 result; // rax

  v10 = a7 != 0;
  result = v10 + (a9 != 0) + (_DWORD)a4 + (a10 != 0) + (a8 != 0) - ((a6 == 0) - 1) - ((unsigned int)(a5 == 0) - 1);
  if ( (int)(v10 + (a9 != 0) + a4 + (a10 != 0) + (a8 != 0) - ((a6 == 0) - 1) - ((a5 == 0) - 1)) <= 0 )
  {
    *(_QWORD *)(a1 + 48) = 0;
  }
  else if ( (int)result > 1 || a10 || a8 != 0 || a7 != 0 || a9 )
  {
    result = (unsigned __int64)sub_2E7B2B0(a2, a3, a4, a5, a6, a7, a8, a9, a10) | 3;
    *(_QWORD *)(a1 + 48) = result;
  }
  else if ( a5 )
  {
    *(_QWORD *)(a1 + 48) = a5 | 1;
  }
  else if ( a6 )
  {
    *(_QWORD *)(a1 + 48) = a6 | 2;
  }
  else
  {
    result = *a3;
    *(_QWORD *)(a1 + 48) = *a3;
  }
  return result;
}
