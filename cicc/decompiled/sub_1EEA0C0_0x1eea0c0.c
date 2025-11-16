// Function: sub_1EEA0C0
// Address: 0x1eea0c0
//
__int64 __fastcall sub_1EEA0C0(__int64 *a1, _QWORD *a2, unsigned int a3)
{
  __int64 v5; // rcx
  unsigned int v6; // esi
  __int64 result; // rax
  _WORD *v8; // r8
  _WORD *v9; // rsi
  unsigned __int64 v10; // rcx
  _WORD *v11; // rdx

  v5 = *a1;
  if ( !*a1 )
    BUG();
  v6 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * a3 + 16);
  result = v6 & 0xF;
  v8 = (_WORD *)(*(_QWORD *)(v5 + 56) + 2LL * (v6 >> 4));
  v10 = a3 * (v6 & 0xF);
  v9 = v8 + 1;
  LOWORD(v10) = *v8 + v10;
  while ( 1 )
  {
    v11 = v9;
    if ( !v9 )
      break;
    while ( 1 )
    {
      ++v11;
      *(_QWORD *)(*a2 + ((v10 >> 3) & 0x1FF8)) |= 1LL << v10;
      result = (unsigned __int16)*(v11 - 1);
      v9 = 0;
      if ( !(_WORD)result )
        break;
      v10 = (unsigned int)(result + v10);
      if ( !v11 )
        return result;
    }
  }
  return result;
}
