// Function: sub_302EA40
// Address: 0x302ea40
//
bool __fastcall sub_302EA40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6, int a7)
{
  bool result; // al

  result = a7 == 1 && *(_WORD *)(*(_QWORD *)(a4 + 48) + 16LL * a5) == 9;
  if ( result )
  {
    *(_QWORD *)a6 = a4;
    *(_DWORD *)(a6 + 8) = a5;
  }
  return result;
}
