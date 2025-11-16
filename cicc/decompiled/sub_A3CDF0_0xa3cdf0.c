// Function: sub_A3CDF0
// Address: 0xa3cdf0
//
__int64 __fastcall sub_A3CDF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  v2 = sub_22077B0(192);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4F8090C;
    *(_QWORD *)(v2 + 56) = v2 + 104;
    *(_QWORD *)(v2 + 112) = v2 + 160;
    *(_DWORD *)(v2 + 24) = 4;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 64) = 1;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_QWORD *)(v2 + 104) = 0;
    *(_QWORD *)(v2 + 120) = 1;
    *(_QWORD *)(v2 + 128) = 0;
    *(_QWORD *)(v2 + 136) = 0;
    *(_QWORD *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 160) = 0;
    *(_BYTE *)(v2 + 168) = 0;
    *(_QWORD *)v2 = off_49D9938;
    *(_QWORD *)(v2 + 176) = a1;
    *(_BYTE *)(v2 + 184) = a2;
    *(_DWORD *)(v2 + 88) = 1065353216;
    *(_DWORD *)(v2 + 144) = 1065353216;
    v4 = sub_BC2B00(192, a2);
    sub_A3CC40(v4);
  }
  return v3;
}
