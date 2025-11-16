// Function: sub_27CDF10
// Address: 0x27cdf10
//
__int64 __fastcall sub_27CDF10(__int64 a1, unsigned __int8 *a2, int a3)
{
  __int64 v4; // rax
  int v5; // eax
  int v6; // edi
  unsigned int v7; // r8d
  int v9; // r10d
  int v10; // edi
  unsigned __int8 *v11; // rsi

  while ( 1 )
  {
    v4 = *((_QWORD *)a2 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
      v4 = **(_QWORD **)(v4 + 16);
    v5 = *(_DWORD *)(v4 + 8) >> 8;
    if ( a3 == v5 )
      return 1;
    v6 = *a2;
    v7 = v6 - 12;
    if ( (unsigned int)(v6 - 12) <= 1 )
      return 1;
    v9 = *(_DWORD *)(a1 + 40);
    if ( v9 != a3 && v9 != v5 )
      return 0;
    if ( (_BYTE)v6 == 20 )
      return 1;
    LOBYTE(v7) = (unsigned __int8)v6 > 0x1Cu || (_BYTE)v6 == 5;
    if ( !(_BYTE)v7 )
      return v7;
    v10 = v6 - 29;
    if ( *a2 <= 0x1Cu )
      v10 = *((unsigned __int16 *)a2 + 1);
    if ( v10 != 50 )
      break;
    if ( (a2[7] & 0x40) != 0 )
      v11 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v11 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    a2 = *(unsigned __int8 **)v11;
  }
  v7 = 0;
  if ( v10 == 48 )
    LOBYTE(v7) = v9 == v5;
  return v7;
}
