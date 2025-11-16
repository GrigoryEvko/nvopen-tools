// Function: sub_1ECE190
// Address: 0x1ece190
//
__int64 __fastcall sub_1ECE190(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 (__fastcall **a5)(__int64, __int64))
{
  __int64 v5; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  char v9; // r8
  __int64 result; // rax

  v5 = a2;
  v7 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v8 = a1 + 24 * v7;
      v9 = (*a5)(v8, a4);
      result = a1 + 24 * v5;
      if ( !v9 )
        break;
      v5 = v7;
      *(_QWORD *)(result + 16) = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(result + 8) = *(_QWORD *)(v8 + 8);
      *(_DWORD *)result = *(_DWORD *)v8;
      if ( a3 >= v7 )
      {
        result = a1 + 24 * v7;
        break;
      }
      v7 = (v7 - 1) / 2;
    }
  }
  else
  {
    result = a1 + 24 * a2;
  }
  *(_QWORD *)(result + 16) = *(_QWORD *)(a4 + 16);
  *(_QWORD *)(result + 8) = *(_QWORD *)(a4 + 8);
  *(_DWORD *)result = *(_DWORD *)a4;
  return result;
}
