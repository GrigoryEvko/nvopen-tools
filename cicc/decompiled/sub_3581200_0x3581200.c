// Function: sub_3581200
// Address: 0x3581200
//
__int64 __fastcall sub_3581200(int a1)
{
  __int64 result; // rax
  int v2; // edx

  result = sub_22077B0(0xE0u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_DWORD *)(result + 24) = 2;
    *(_QWORD *)(result + 16) = &unk_503F16C;
    *(_QWORD *)(result + 56) = result + 104;
    *(_QWORD *)(result + 112) = result + 160;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)result = &unk_4A39748;
    v2 = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_QWORD *)(result + 64) = 1;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 80) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 120) = 1;
    *(_QWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_QWORD *)(result + 160) = 0;
    *(_BYTE *)(result + 168) = 0;
    *(_QWORD *)(result + 176) = 0;
    *(_QWORD *)(result + 184) = 0;
    *(_QWORD *)(result + 192) = 0;
    *(_QWORD *)(result + 200) = 0;
    *(_DWORD *)(result + 208) = a1;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
    if ( a1 )
      v2 = 2 * (3 * a1 - 3) + 8;
    *(_DWORD *)(result + 212) = v2;
    *(_DWORD *)(result + 216) = 6 * a1 + 7;
  }
  return result;
}
