// Function: sub_2957D10
// Address: 0x2957d10
//
__int64 __fastcall sub_2957D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 result; // rax

  v4 = a3 + 48;
  v5 = *(_QWORD *)(a3 + 56);
  if ( v5 == a3 + 48 )
LABEL_15:
    BUG();
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    if ( *(_BYTE *)(v5 - 24) != 84 )
      return 1;
    v7 = *(_QWORD *)(v5 - 32);
    v8 = 0x1FFFFFFFE0LL;
    v9 = *(_DWORD *)(v5 - 20) & 0x7FFFFFF;
    if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) != 0 )
    {
      v10 = 0;
      a4 = v7 + 32LL * *(unsigned int *)(v5 + 48);
      do
      {
        if ( a2 == *(_QWORD *)(a4 + 8 * v10) )
        {
          v8 = 32 * v10;
          goto LABEL_9;
        }
        ++v10;
      }
      while ( (_DWORD)v9 != (_DWORD)v10 );
      v8 = 0x1FFFFFFFE0LL;
    }
LABEL_9:
    result = sub_D48480(a1, *(_QWORD *)(v7 + v8), v9, a4);
    if ( !(_BYTE)result )
      return result;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v4 == v5 )
      goto LABEL_15;
  }
}
