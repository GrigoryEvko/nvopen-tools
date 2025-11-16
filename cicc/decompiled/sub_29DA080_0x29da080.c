// Function: sub_29DA080
// Address: 0x29da080
//
__int64 __fastcall sub_29DA080(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rdx
  __int64 result; // rax
  unsigned __int64 i; // r13
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned __int8 v11; // al

  if ( a2 == a3 )
    return 0;
  if ( !a2 )
    return 0xFFFFFFFFLL;
  if ( !a3 )
    return 1;
  if ( (*(_BYTE *)(a3 - 16) & 2) != 0 )
  {
    v5 = *(unsigned int *)(a3 - 24);
    if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
      goto LABEL_6;
LABEL_18:
    result = sub_29D7CF0((__int64)a1, (*(_WORD *)(a2 - 16) >> 6) & 0xF, v5);
    if ( !(_DWORD)result )
      goto LABEL_7;
    return result;
  }
  v5 = (*(_WORD *)(a3 - 16) >> 6) & 0xF;
  if ( (*(_BYTE *)(a2 - 16) & 2) == 0 )
    goto LABEL_18;
LABEL_6:
  result = sub_29D7CF0((__int64)a1, *(unsigned int *)(a2 - 24), v5);
  if ( !(_DWORD)result )
  {
LABEL_7:
    for ( i = 0; ; ++i )
    {
      if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
      {
        if ( *(unsigned int *)(a2 - 24) <= i )
          return 0;
      }
      else if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) <= i )
      {
        return 0;
      }
      v11 = *(_BYTE *)(a3 - 16);
      if ( (v11 & 2) != 0 )
      {
        v8 = 8 * i;
        v9 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 8 * i);
        if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
          goto LABEL_9;
      }
      else
      {
        v8 = 8 * i;
        v9 = *(_QWORD *)(a3 + -16 - 8LL * ((v11 >> 2) & 0xF) + 8 * i);
        if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
        {
LABEL_9:
          v10 = *(_QWORD *)(a2 - 32);
          goto LABEL_10;
        }
      }
      v10 = a2 + -16 - 8LL * ((*(_BYTE *)(a2 - 16) >> 2) & 0xF);
LABEL_10:
      result = sub_29D9F90(a1, *(_QWORD *)(v10 + v8), v9);
      if ( (_DWORD)result )
        return result;
    }
  }
  return result;
}
