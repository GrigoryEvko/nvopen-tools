// Function: sub_216FEC0
// Address: 0x216fec0
//
__int64 __fastcall sub_216FEC0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int16 v3; // ax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // edx
  char v14; // di

  v3 = *(_WORD *)(a1 + 24);
  if ( v3 == -3244 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL) + 88LL);
    v12 = *(_QWORD **)(v11 + 24);
    if ( *(_DWORD *)(v11 + 32) > 0x40u )
      v12 = (_QWORD *)*v12;
    *a2 = 0;
    v13 = 0;
    v14 = 0;
    *a3 = 0;
    if ( ((unsigned __int8)v12 & 0xC) != 0 )
      goto LABEL_16;
LABEL_13:
    *a3 += 8LL;
    v14 = 1;
    while ( 1 )
    {
      ++v13;
      LODWORD(v12) = (unsigned int)v12 >> 4;
      if ( v13 == 4 )
        break;
      if ( ((unsigned __int8)v12 & 0xC) == 0 )
        goto LABEL_13;
LABEL_16:
      if ( v14 )
      {
        while ( 1 )
        {
          ++v13;
          LODWORD(v12) = (unsigned int)v12 >> 4;
          if ( v13 == 4 )
            break;
          if ( ((unsigned __int8)v12 & 0xC) == 0 )
            return 0;
        }
        return 1;
      }
      *a2 += 8LL;
    }
  }
  else
  {
    if ( v3 != -166 && v3 != -165 )
      return 0;
    v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL) + 88LL);
    v7 = *(_QWORD **)(v6 + 24);
    if ( *(_DWORD *)(v6 + 32) > 0x40u )
      v7 = (_QWORD *)*v7;
    *a2 = v7;
    v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 120LL) + 88LL);
    v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
    v10 = *(_QWORD **)(v8 + 24);
    if ( !v9 )
      v10 = (_QWORD *)*v10;
    *a3 = v10;
  }
  return 1;
}
