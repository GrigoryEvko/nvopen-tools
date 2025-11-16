// Function: sub_1440FD0
// Address: 0x1440fd0
//
__int64 __fastcall sub_1440FD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  v2 = sub_22077B0(264);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 24) = 3;
    *(_QWORD *)(v2 + 16) = &unk_4F99CCC;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)v2 = &unk_49EB980;
    *(_QWORD *)(v2 + 160) = v2 + 176;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 168) = 0x400000000LL;
    *(_QWORD *)(v2 + 208) = 0;
    *(_QWORD *)(v2 + 216) = 0;
    *(_QWORD *)(v2 + 224) = 0;
    *(_DWORD *)(v2 + 232) = 0;
    *(_QWORD *)(v2 + 248) = 0;
    *(_BYTE *)(v2 + 256) = 0;
    *(_DWORD *)(v2 + 260) = 0;
    v4 = sub_163A1D0(264, a2);
    sub_1440EE0(v4);
  }
  return v3;
}
