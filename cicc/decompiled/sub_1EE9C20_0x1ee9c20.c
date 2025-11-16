// Function: sub_1EE9C20
// Address: 0x1ee9c20
//
__int64 __fastcall sub_1EE9C20(__int64 a1, unsigned int a2, int a3)
{
  _QWORD *v3; // r11
  __int64 v6; // r10
  unsigned int v7; // edx
  __int16 v8; // ax
  _WORD *v9; // rdx
  __int16 v10; // cx
  __int64 result; // rax
  unsigned __int16 *v12; // rdi
  unsigned __int16 v13; // cx
  _DWORD *v14; // rsi
  unsigned __int16 *v15; // rdx

  v3 = *(_QWORD **)(a1 + 96);
  if ( !v3 )
    BUG();
  v6 = v3[1] + 24LL * a2;
  v7 = *(_DWORD *)(v6 + 16);
  v8 = v7 & 0xF;
  v9 = (_WORD *)(v3[7] + 2LL * (v7 >> 4));
  v10 = v8;
  result = v3[8];
  v12 = v9 + 1;
  v13 = *v9 + a2 * v10;
  v14 = (_DWORD *)(result + 4LL * *(unsigned __int16 *)(v6 + 20));
  while ( 1 )
  {
    v15 = v12;
    if ( !v12 )
      break;
    while ( 1 )
    {
      if ( !*v14 || (*v14 & a3) != 0 )
        *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (v13 >> 6)) |= 1LL << v13;
      result = *v15;
      ++v14;
      ++v15;
      v12 = 0;
      if ( !(_WORD)result )
        break;
      v13 += result;
      if ( !v15 )
        return result;
    }
  }
  return result;
}
