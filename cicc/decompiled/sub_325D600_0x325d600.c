// Function: sub_325D600
// Address: 0x325d600
//
char __fastcall sub_325D600(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char result; // al
  __int64 v7; // rdx
  __int64 v8; // rdx

  result = sub_33DFCF0(a1, a2, 0);
  if ( !result || (v7 = *(_QWORD *)(a1 + 40), *(_QWORD *)v7 != a3) || *(_DWORD *)(v7 + 8) != (_DWORD)a4 )
  {
    result = sub_33DFCF0(a3, a4, 0);
    if ( result )
    {
      v8 = *(_QWORD *)(a3 + 40);
      result = 0;
      if ( *(_QWORD *)v8 == a1 )
        return *(_DWORD *)(v8 + 8) == (_DWORD)a2;
    }
  }
  return result;
}
