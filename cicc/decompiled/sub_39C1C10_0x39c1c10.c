// Function: sub_39C1C10
// Address: 0x39c1c10
//
__int64 __fastcall sub_39C1C10(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rdx

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    v7 = a1 + 8;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(result + 16);
        v9 = *(_QWORD *)(result + 24);
        if ( *(_DWORD *)(result + 32) >= a2 )
          break;
        result = *(_QWORD *)(result + 24);
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = result;
      result = *(_QWORD *)(result + 16);
    }
    while ( v8 );
LABEL_6:
    if ( a1 + 8 != v7 && *(_DWORD *)(v7 + 32) <= a2 )
      return sub_39C1B70(a1, v7, a3, a4);
  }
  return result;
}
