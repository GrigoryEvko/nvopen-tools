// Function: sub_20D6CE0
// Address: 0x20d6ce0
//
__int64 __fastcall sub_20D6CE0(__int64 a1, char a2, char a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = a1 + 64;
  *(_QWORD *)(a1 + 40) = a1 + 64;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 2;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_BYTE *)(a1 + 138) = a3;
  *(_DWORD *)(a1 + 140) = a6;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 200) = 0x800000000LL;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = a4;
  *(_QWORD *)(a1 + 264) = a5;
  if ( !a6 )
    *(_DWORD *)(a1 + 140) = dword_4FCF3C0;
  result = (unsigned int)dword_4FCF580;
  switch ( dword_4FCF580 )
  {
    case 1:
      *(_BYTE *)(a1 + 137) = 1;
      break;
    case 2:
      *(_BYTE *)(a1 + 137) = 0;
      break;
    case 0:
      *(_BYTE *)(a1 + 137) = a2;
      break;
  }
  return result;
}
