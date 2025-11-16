// Function: sub_307AC10
// Address: 0x307ac10
//
__int64 __fastcall sub_307AC10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  v2 = sub_22077B0(0xF0u);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 24) = 2;
    *(_QWORD *)(v2 + 16) = &unk_502D264;
    *(_QWORD *)(v2 + 56) = v2 + 104;
    *(_QWORD *)(v2 + 112) = v2 + 160;
    *(_QWORD *)(v2 + 32) = 0;
    *(_DWORD *)(v2 + 88) = 1065353216;
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
    *(_QWORD *)(v2 + 176) = 0;
    *(_QWORD *)(v2 + 184) = 0;
    *(_QWORD *)(v2 + 192) = 0;
    *(_QWORD *)v2 = &unk_4A31850;
    *(_DWORD *)(v2 + 144) = 1065353216;
    v4 = sub_C5F790(240, a2);
    *(_QWORD *)(v3 + 216) = 0;
    *(_QWORD *)(v3 + 200) = v4;
    *(_QWORD *)(v3 + 208) = v3 + 224;
    *(_BYTE *)(v3 + 224) = 0;
  }
  return v3;
}
