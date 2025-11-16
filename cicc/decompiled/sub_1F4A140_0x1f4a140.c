// Function: sub_1F4A140
// Address: 0x1f4a140
//
__int64 __fastcall sub_1F4A140(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // r11
  unsigned __int16 *v7; // rcx
  __int64 v8; // rsi
  int v9; // edx
  unsigned __int16 *v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rdi
  __int64 v13; // rsi
  unsigned __int16 *v14; // rax
  __int64 v16; // rdx
  unsigned int v17; // edi

  v6 = *(_QWORD *)(a1 + 256);
  v7 = *(unsigned __int16 **)(a3 + 16);
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *v7;
  v10 = v7 + 1;
  v11 = (*(_QWORD *)(a1 + 264) - v6) >> 3;
  v12 = 4LL * ((unsigned int)(v11 + 31) >> 5);
  v13 = v12 + v8;
  if ( !v9 )
    v10 = 0;
LABEL_6:
  v14 = v10;
  while ( 1 )
  {
    if ( !v14 )
      return 0;
    if ( a4 == v9 )
      break;
    v9 = *v14;
    v13 += v12;
    ++v14;
    v10 = 0;
    if ( !v9 )
      goto LABEL_6;
  }
  if ( !(_DWORD)v11 )
    return 0;
  v16 = 0;
  v17 = 0;
  while ( !(*(_DWORD *)(*(_QWORD *)(a2 + 8) + v16) & *(_DWORD *)(v13 + v16)) )
  {
    v17 += 32;
    v16 += 4;
    if ( (unsigned int)v11 <= v17 )
      return 0;
  }
  __asm { tzcnt   eax, eax }
  return *(_QWORD *)(v6 + 8LL * (v17 + _EAX));
}
