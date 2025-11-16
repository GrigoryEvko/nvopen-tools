// Function: sub_20D60E0
// Address: 0x20d60e0
//
__int64 __fastcall sub_20D60E0(__int64 a1)
{
  int v1; // r9d
  unsigned int v2; // r8d
  unsigned __int8 *v3; // rdx
  int i; // ecx
  int v5; // eax
  int v6; // esi
  int v7; // eax

  v1 = *(_DWORD *)(a1 + 40);
  v2 = **(unsigned __int16 **)(a1 + 16);
  if ( v1 )
  {
    v3 = *(unsigned __int8 **)(a1 + 32);
    for ( i = 0; i != v1; ++i )
    {
      v5 = *v3;
      switch ( (char)v5 )
      {
        case 0:
          v6 = 8 * *((_DWORD *)v3 + 2);
          break;
        case 1:
        case 5:
        case 6:
        case 8:
          v6 = 8 * *((_DWORD *)v3 + 6);
          break;
        case 4:
          v6 = 8 * *(_DWORD *)(*((_QWORD *)v3 + 3) + 48LL);
          break;
        case 9:
        case 10:
          v6 = 8 * *((_DWORD *)v3 + 2);
          break;
        default:
          v6 = 0;
          break;
      }
      v3 += 40;
      v7 = (v6 | v5) << i;
      v2 += v7;
    }
  }
  return v2;
}
