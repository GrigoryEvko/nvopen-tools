// Function: sub_1F4AF90
// Address: 0x1f4af90
//
__int64 __fastcall sub_1F4AF90(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rbx
  __int64 v6; // r11
  __int64 v7; // r8
  unsigned int v8; // r10d
  __int64 v9; // rsi
  unsigned int v10; // edi
  __int64 v13; // r12
  char *v14; // rdx
  char v15; // al

  if ( a2 == a3 )
    return a2;
  if ( !a2 )
    return 0;
  if ( !a3 )
    return 0;
  v4 = *(_QWORD *)(a1 + 256);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a3 + 8);
  v8 = (*(_QWORD *)(a1 + 264) - v4) >> 3;
  if ( !v8 )
    return 0;
  v9 = 0;
  v10 = 0;
  while ( 1 )
  {
    if ( *(_DWORD *)(v7 + v9) & *(_DWORD *)(v6 + v9) )
    {
      __asm { tzcnt   eax, eax }
      v13 = *(_QWORD *)(v4 + 8LL * (v10 + _EAX));
      if ( a4 == -1 )
        return v13;
      v14 = *(char **)(*(_QWORD *)(a1 + 280)
                     + 24LL * (v8 * *(_DWORD *)(a1 + 288) + *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL))
                     + 16);
      v15 = *v14;
      if ( *v14 != 1 )
        break;
    }
LABEL_13:
    v10 += 32;
    v9 += 4;
    if ( v8 <= v10 )
      return 0;
  }
  while ( a4 != v15 )
  {
    v15 = *++v14;
    if ( v15 == 1 )
      goto LABEL_13;
  }
  return v13;
}
