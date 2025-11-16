// Function: sub_890680
// Address: 0x890680
//
_BOOL8 __fastcall sub_890680(__int64 a1, char a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rdi
  _QWORD *v7; // rax
  bool v8; // cf
  __int64 v9; // rcx
  __int128 *v10; // rsi
  _BOOL8 result; // rax

  if ( dword_4F04C44 == -1 )
    return 1;
  v3 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 408);
  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v4 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v5 = 0;
  v6 = *(_QWORD *)(*(_QWORD *)(v4 + 104) + 176LL);
  v7 = (_QWORD *)v6;
  do
  {
    v8 = v7[1] == 0;
    v7 = (_QWORD *)*v7;
    v5 -= v8 - 1;
  }
  while ( v7 );
  v9 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 408);
  if ( !v3 )
    goto LABEL_10;
  do
  {
    v9 = *(_QWORD *)(v9 + 24);
    LODWORD(v7) = (_DWORD)v7 + 1;
  }
  while ( v9 );
  while ( v5 != (_DWORD)v7 )
  {
    v3 = *(_QWORD *)(v3 + 24);
    LODWORD(v7) = (_DWORD)v7 - 1;
LABEL_10:
    ;
  }
  v10 = *(__int128 **)(*(_QWORD *)(v3 + 32) + 16LL);
  if ( v10 )
    return (unsigned int)sub_739400(*(__int128 **)(v6 + 16), v10) != 0;
  result = 1;
  if ( (a2 & 1) == 0 )
    return (unsigned int)sub_739400(*(__int128 **)(v6 + 16), v10) != 0;
  return result;
}
