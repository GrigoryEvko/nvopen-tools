// Function: sub_37F5910
// Address: 0x37f5910
//
__int64 __fastcall sub_37F5910(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rsi
  __int64 v6; // r10
  __int64 v7; // r8
  int v8; // r11d
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // rbx
  int v12; // edx
  int v13; // r12d

  if ( a3 < 0 )
    return 0;
  result = *(_QWORD *)(a2 + 56);
  v4 = a2 + 48;
  if ( result == v4 )
    return 0;
  v6 = *(_QWORD *)(a1 + 472);
  v7 = *(unsigned int *)(a1 + 488);
  v8 = v7 - 1;
  do
  {
    while ( 1 )
    {
      if ( (_DWORD)v7 )
      {
        v9 = v8 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( result == *v10 )
        {
LABEL_7:
          if ( (__int64 *)(v6 + 16 * v7) != v10 && *((_DWORD *)v10 + 2) == a3 )
            return result;
        }
        else
        {
          v12 = 1;
          while ( v11 != -4096 )
          {
            v13 = v12 + 1;
            v9 = v8 & (v12 + v9);
            v10 = (__int64 *)(v6 + 16LL * v9);
            v11 = *v10;
            if ( *v10 == result )
              goto LABEL_7;
            v12 = v13;
          }
        }
      }
      if ( !result )
        BUG();
      if ( (*(_BYTE *)result & 4) == 0 )
        break;
      result = *(_QWORD *)(result + 8);
      if ( v4 == result )
        return 0;
    }
    while ( (*(_BYTE *)(result + 44) & 8) != 0 )
      result = *(_QWORD *)(result + 8);
    result = *(_QWORD *)(result + 8);
  }
  while ( v4 != result );
  return 0;
}
