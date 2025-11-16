// Function: sub_2D285B0
// Address: 0x2d285b0
//
__int64 __fastcall sub_2D285B0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rdx

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 16) = &unk_50165D0;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)a1 = &unk_4A262E8;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  result = sub_22077B0(0x90u);
  v2 = (_QWORD *)result;
  if ( result )
  {
    memset((void *)result, 0, 0x90u);
    *(_QWORD *)result = result + 16;
    result = 0x100000000LL;
    v2[1] = 0x100000000LL;
    v2[7] = v2 + 9;
    v2[8] = 0x100000000LL;
  }
  *(_QWORD *)(a1 + 176) = v2;
  return result;
}
