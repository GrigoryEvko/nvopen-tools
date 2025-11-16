// Function: sub_88DC10
// Address: 0x88dc10
//
__int64 __fastcall sub_88DC10(__int64 a1, FILE *a2, __int64 a3)
{
  char v3; // r8
  int v5; // edx
  __int64 i; // rax
  __int64 result; // rax
  unsigned __int8 v8; // r9

  v3 = *(_BYTE *)(a1 + 80);
  switch ( v3 )
  {
    case 4:
    case 5:
      v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 168LL) + 168LL) != 0;
      break;
    case 6:
    case 9:
      v5 = 0;
      break;
    case 7:
    case 19:
    case 20:
    case 21:
      v5 = 1;
      break;
    case 10:
    case 11:
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 240LL) != 0;
      break;
    default:
      sub_721090();
  }
  if ( (!*(_DWORD *)(a3 + 20) || *(_DWORD *)(a3 + 16)) && (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    for ( i = *(_QWORD *)(a1 + 64); i; i = *(_QWORD *)(*(_QWORD *)(i + 40) + 32LL) )
    {
      while ( *(_BYTE *)(i + 140) == 12 )
        i = *(_QWORD *)(i + 160);
      if ( (*(_BYTE *)(i + 178) & 1) != 0 )
        break;
      v5 -= (*(_QWORD *)(*(_QWORD *)(i + 168) + 168LL) == 0) - 1;
      if ( (*(_BYTE *)(i + 89) & 4) == 0 )
        break;
    }
  }
  result = *(_QWORD *)(a3 + 224) + *(int *)(a3 + 172);
  if ( v5 != result && !*(_DWORD *)(a3 + 52) && !*(_DWORD *)(a3 + 108) )
  {
    v8 = 7;
    if ( dword_4F077BC )
    {
      if ( !*(_DWORD *)(a3 + 16) || (v8 = 5, v3 != 19) )
        v8 = *(_DWORD *)(a3 + 32) == 0 ? 7 : 5;
    }
    result = sub_6853B0(v8, 0x308u, a2, a1);
    *(_DWORD *)(a3 + 56) = 1;
  }
  return result;
}
