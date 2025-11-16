// Function: sub_36CFC80
// Address: 0x36cfc80
//
__int64 __fastcall sub_36CFC80(__int64 a1, __int64 a2, unsigned __int8 **a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  unsigned __int8 *v7; // rax
  int v8; // eax
  __int64 v9; // rcx
  _QWORD *v10; // rax
  int v11; // ebx
  unsigned int v12; // eax
  _BYTE *v13; // rdi

  if ( !(_BYTE)qword_50409C8 )
    return 3;
  if ( *(_BYTE *)a2 != 85 )
    return 3;
  v7 = *(unsigned __int8 **)(a2 - 32);
  if ( !v7 )
    return 3;
  v5 = *v7;
  if ( (_BYTE)v5 || *((_QWORD *)v7 + 3) != *(_QWORD *)(a2 + 80) || (v7[33] & 0x20) == 0 )
    return 3;
  v8 = *((_DWORD *)v7 + 9);
  if ( v8 == 8604 )
  {
    v13 = *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v13 == 17 && sub_AD7890((__int64)v13, a2, (__int64)a3, a4, a5) )
      return v5;
    return 3;
  }
  if ( v8 != 8605 )
    return 3;
  v9 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v9 != 17 )
    return 3;
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  if ( (unsigned __int64)v10 > 3 )
  {
    if ( v10 != (_QWORD *)4 )
      return 3;
    v11 = 3;
  }
  else
  {
    v11 = 2;
    if ( (unsigned __int64)v10 <= 1 )
      v11 = v10 == 0 ? 3 : 1;
  }
  v12 = sub_36CDC90(*a3, qword_5040B88);
  if ( v12 <= 6 )
  {
    if ( v12 )
    {
      switch ( v12 )
      {
        case 1u:
          goto LABEL_21;
        case 3u:
          v12 = 2;
LABEL_21:
          if ( (v11 & v12) != 0 )
            v5 = 3;
          break;
        case 4u:
        case 5u:
        case 6u:
          return v5;
        default:
          return 3;
      }
      return v5;
    }
    return 3;
  }
  if ( v12 != 101 )
    return 3;
  return v5;
}
