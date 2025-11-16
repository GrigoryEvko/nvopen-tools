// Function: sub_1A4F070
// Address: 0x1a4f070
//
__int64 __fastcall sub_1A4F070(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi

  result = sub_157F280(a1);
  v5 = v4;
  if ( result != v4 )
  {
    v6 = result;
    do
    {
      v7 = 0;
      v8 = 8LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
      if ( (*(_DWORD *)(v6 + 20) & 0xFFFFFFF) != 0 )
      {
        do
        {
          if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
            v9 = *(_QWORD *)(v6 - 8);
          else
            v9 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
          v7 += 8;
          *(_QWORD *)(v7 + v9 + 24LL * *(unsigned int *)(v6 + 56)) = a2;
        }
        while ( v8 != v7 );
      }
      result = *(_QWORD *)(v6 + 32);
      if ( !result )
        BUG();
      v6 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v6 = result - 24;
    }
    while ( v5 != v6 );
  }
  return result;
}
