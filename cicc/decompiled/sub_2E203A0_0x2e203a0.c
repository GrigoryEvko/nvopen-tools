// Function: sub_2E203A0
// Address: 0x2e203a0
//
__int64 sub_2E203A0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rdx

  v0 = sub_22077B0(0x158u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 16) = &unk_501EAFC;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)(v0 + 32) = 0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    *(_BYTE *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)v0 = &unk_4A284E0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_DWORD *)(v0 + 224) = 0;
    *(_DWORD *)(v0 + 144) = 1065353216;
    v2 = (_QWORD *)sub_22077B0(0x68u);
    if ( v2 )
    {
      memset(v2, 0, 0x68u);
      v2[12] = 1;
      v2[3] = v2 + 5;
      v2[4] = 0x400000000LL;
      v2[9] = v2 + 11;
    }
    *(_QWORD *)(v1 + 232) = v2;
    *(_QWORD *)(v1 + 272) = v1 + 288;
    *(_DWORD *)(v1 + 240) = 0;
    *(_QWORD *)(v1 + 248) = 0;
    *(_QWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 264) = 0;
    *(_QWORD *)(v1 + 280) = 0x600000000LL;
    *(_DWORD *)(v1 + 336) = 0;
  }
  return v1;
}
