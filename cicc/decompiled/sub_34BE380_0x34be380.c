// Function: sub_34BE380
// Address: 0x34be380
//
__int64 __fastcall sub_34BE380(__int64 a1)
{
  unsigned int v1; // r8d
  unsigned __int8 *v2; // rdx
  int v3; // ecx
  int v4; // eax
  int v5; // esi
  int v6; // eax

  v1 = *(unsigned __int16 *)(a1 + 68);
  if ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != 0 )
  {
    v2 = *(unsigned __int8 **)(a1 + 32);
    v3 = 0;
    do
    {
      v4 = *v2;
      switch ( (char)v4 )
      {
        case 0:
          v5 = 8 * *((_DWORD *)v2 + 2);
          break;
        case 1:
        case 5:
        case 6:
        case 8:
          v5 = 8 * *((_DWORD *)v2 + 6);
          break;
        case 4:
          v5 = 8 * *(_DWORD *)(*((_QWORD *)v2 + 3) + 24LL);
          break;
        case 9:
        case 10:
          v5 = 8 * *((_DWORD *)v2 + 2);
          break;
        default:
          v5 = 0;
          break;
      }
      v2 += 40;
      v6 = (v5 | v4) << v3++;
      v1 += v6;
    }
    while ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != v3 );
  }
  return v1;
}
