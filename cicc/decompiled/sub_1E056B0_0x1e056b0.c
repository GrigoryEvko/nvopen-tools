// Function: sub_1E056B0
// Address: 0x1e056b0
//
void __fastcall sub_1E056B0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  _QWORD *v4; // rax
  __int64 v5; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_4FC62EC;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  v2 = a1 + 112;
  v3 = a1 + 160;
  *(_QWORD *)(v3 - 32) = v2;
  *(_QWORD *)(v3 - 24) = v2;
  *(_QWORD *)(v3 - 128) = 0;
  *(_QWORD *)(v3 - 120) = 0;
  *(_QWORD *)(v3 - 160) = &unk_49FB790;
  *(_QWORD *)(v3 - 112) = 0;
  *(_DWORD *)(v3 - 96) = 0;
  *(_QWORD *)(v3 - 88) = 0;
  *(_QWORD *)(v3 - 64) = 0;
  *(_DWORD *)(v3 - 48) = 0;
  *(_QWORD *)(v3 - 40) = 0;
  *(_QWORD *)(v3 - 16) = 0;
  *(_BYTE *)(v3 - 8) = 0;
  sub_1BFC1A0(v3, 8, 0);
  sub_1BFC1A0(a1 + 184, 8, 0);
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v4 = (_QWORD *)malloc(8u);
  if ( !v4 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v4 = 0;
  }
  *(_QWORD *)(a1 + 208) = v4;
  *v4 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_QWORD *)a1 = &unk_49FB698;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 240) = 0x2000000000LL;
  *(_QWORD *)(a1 + 1024) = a1 + 1056;
  *(_QWORD *)(a1 + 1032) = a1 + 1056;
  *(_QWORD *)(a1 + 1040) = 32;
  *(_DWORD *)(a1 + 1048) = 0;
  *(_QWORD *)(a1 + 1312) = 0;
  v5 = sub_163A1D0();
  sub_1E055C0(v5);
}
