// Function: sub_3598310
// Address: 0x3598310
//
__int64 __fastcall sub_3598310(__int64 a1, __int64 a2, _DWORD *a3, _DWORD *a4)
{
  __int64 result; // rax
  int v8; // edi
  __int64 v9; // rdx
  int v10; // esi

  *a3 = 0;
  *a4 = 0;
  result = 1;
  v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v8 != 1 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(a1 + 32);
        v10 = *(_DWORD *)(v9 + 40LL * (unsigned int)result + 8);
        if ( *(_QWORD *)(v9 + 40LL * (unsigned int)(result + 1) + 24) == a2 )
          break;
        result = (unsigned int)(result + 2);
        *a3 = v10;
        if ( v8 == (_DWORD)result )
          return result;
      }
      result = (unsigned int)(result + 2);
      *a4 = v10;
    }
    while ( v8 != (_DWORD)result );
  }
  return result;
}
