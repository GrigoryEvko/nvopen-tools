// Function: sub_2E30860
// Address: 0x2e30860
//
__int64 __fastcall sub_2E30860(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx

  *(_QWORD *)(a1 + 24) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 56) = a1 + 48;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0x400000000LL;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 48) = (a1 + 48) | 4;
  *(_QWORD *)(a1 + 120) = 0x200000000LL;
  result = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_BYTE *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 212) = 0;
  *(_WORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_BYTE *)(a1 + 236) = 0;
  *(_BYTE *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 252) = 0;
  *(_WORD *)(a1 + 260) = 0;
  *(_BYTE *)(a1 + 262) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 40) = a1;
  if ( a3 )
  {
    result = sub_AA5EE0(a3);
    *(_QWORD *)(a1 + 168) = result;
    *(_QWORD *)(a1 + 176) = v4;
  }
  return result;
}
