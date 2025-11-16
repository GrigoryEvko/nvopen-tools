// Function: sub_6E8B30
// Address: 0x6e8b30
//
__int64 __fastcall sub_6E8B30(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi

  for ( result = *a1; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
    ;
  *a3 = result;
  v9 = *a1;
  v10 = *a2;
  if ( *a1 != *a2 )
  {
    result = sub_8D97D0(v9, v10, 32, a4, a5);
    if ( !(_DWORD)result )
    {
      if ( (unsigned int)sub_8D28B0(*a1) )
      {
        return sub_6E68E0(0x85Eu, (__int64)a2);
      }
      else
      {
        sub_6E68E0(0x85Eu, (__int64)a1);
        for ( result = *a2; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
          ;
        *a3 = result;
      }
    }
  }
  return result;
}
