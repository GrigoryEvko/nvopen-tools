// Function: sub_15E9F00
// Address: 0x15e9f00
//
__int64 __fastcall sub_15E9F00(__int64 a1, unsigned __int8 **a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // r12

  v4 = sub_22077B0(208);
  v5 = v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = &unk_4F9E23C;
    *(_QWORD *)(v4 + 80) = v4 + 64;
    *(_QWORD *)(v4 + 88) = v4 + 64;
    *(_QWORD *)(v4 + 128) = v4 + 112;
    *(_QWORD *)(v4 + 136) = v4 + 112;
    *(_DWORD *)(v4 + 24) = 5;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    *(_DWORD *)(v4 + 64) = 0;
    *(_QWORD *)(v4 + 72) = 0;
    *(_QWORD *)(v4 + 96) = 0;
    *(_DWORD *)(v4 + 112) = 0;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 144) = 0;
    *(_BYTE *)(v4 + 152) = 0;
    *(_QWORD *)v4 = off_49ED1D8;
    sub_15E92E0(v4 + 160, a1, a2, a3);
  }
  return v5;
}
