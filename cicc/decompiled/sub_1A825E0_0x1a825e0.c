// Function: sub_1A825E0
// Address: 0x1a825e0
//
__int64 __fastcall sub_1A825E0(__int64 a1, __int64 *a2, int a3, unsigned int a4)
{
  __int64 v4; // rax
  int v6; // edx
  int v7; // eax
  int v8; // r10d
  int v9; // eax
  __int64 *v10; // rsi

  v4 = *a2;
  if ( *(_BYTE *)(*a2 + 8) == 16 )
  {
LABEL_2:
    v4 = **(_QWORD **)(v4 + 16);
    goto LABEL_3;
  }
  while ( 1 )
  {
LABEL_3:
    v6 = *(_DWORD *)(v4 + 8) >> 8;
    if ( a3 == v6 )
      return 1;
    v7 = *((unsigned __int8 *)a2 + 16);
    if ( (_BYTE)v7 == 9 )
      return 1;
    v8 = *(_DWORD *)(a1 + 156);
    if ( v8 != v6 && v8 != a3 )
      return 0;
    if ( (_BYTE)v7 == 15 )
      return 1;
    LOBYTE(a4) = (unsigned __int8)v7 > 0x17u || (_BYTE)v7 == 5;
    if ( !(_BYTE)a4 )
      return a4;
    v9 = (unsigned __int8)v7 <= 0x17u ? *((unsigned __int16 *)a2 + 9) : v7 - 24;
    if ( v9 != 48 )
      break;
    if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
      v10 = (__int64 *)*(a2 - 1);
    else
      v10 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    a2 = (__int64 *)*v10;
    v4 = *a2;
    if ( *(_BYTE *)(*a2 + 8) == 16 )
      goto LABEL_2;
  }
  LOBYTE(a4) = v9 == 46;
  LOBYTE(v9) = v8 == v6;
  a4 &= v9;
  return a4;
}
