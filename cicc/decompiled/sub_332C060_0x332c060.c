// Function: sub_332C060
// Address: 0x332c060
//
__int64 __fastcall sub_332C060(unsigned __int16 *a1, _QWORD *a2)
{
  __int64 v2; // r14
  unsigned __int16 v3; // ax
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rbx
  int v7; // r15d
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rax
  unsigned int v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  char v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  if ( *a1 )
  {
    if ( v3 == 1 || (unsigned __int16)(v3 - 504) <= 7u )
      BUG();
    v5 = 16LL * (v3 - 1);
    v4 = *(_QWORD *)&byte_444C4A0[v5];
    LOBYTE(v5) = byte_444C4A0[v5 + 8];
  }
  else
  {
    v4 = sub_3007260((__int64)a1);
    v16 = v4;
    v17 = v5;
  }
  v14 = v4;
  v6 = &byte_444C4A0[16];
  v7 = 2;
  v15 = v5;
  v13 = sub_CA1930(&v14);
  do
  {
    v8 = 2LL * *(_QWORD *)v6;
    v15 = v6[8];
    v14 = v8;
    if ( sub_CA1930(&v14) >= (unsigned __int64)v13 )
      return (unsigned __int16)v7;
    ++v7;
    v6 += 16;
  }
  while ( v7 != 10 );
  v9 = (v13 + 1) >> 1;
  switch ( v9 )
  {
    case 1u:
      LOWORD(v10) = 2;
      break;
    case 2u:
      LOWORD(v10) = 3;
      break;
    case 4u:
      LOWORD(v10) = 4;
      break;
    case 8u:
      LOWORD(v10) = 5;
      break;
    case 0x10u:
      LOWORD(v10) = 6;
      break;
    case 0x20u:
      LOWORD(v10) = 7;
      break;
    case 0x40u:
      LOWORD(v10) = 8;
      break;
    case 0x80u:
      LOWORD(v10) = 9;
      break;
    default:
      v10 = sub_3007020(a2, v9);
      v2 = v10;
      break;
  }
  LOWORD(v2) = v10;
  return v2;
}
