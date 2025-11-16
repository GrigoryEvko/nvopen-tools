// Function: sub_A18170
// Address: 0xa18170
//
void __fastcall sub_A18170(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi

  v3 = sub_22077B0(152);
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)v3 = v3 + 24;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 24) = a2;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 56) = 2;
    *(_QWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 80) = 0;
    *(_BYTE *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 104) = 0;
    *(_QWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 128) = 0;
    *(_QWORD *)(v3 + 136) = 0;
    *(_QWORD *)(v3 + 144) = 0;
  }
  *(_QWORD *)a1 = v3;
  sub_C0BFB0(a1 + 8, 6, 0);
  *(_QWORD *)(a1 + 56) = 0;
  v4 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 1;
  *(_WORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  sub_A17BC0(v4);
}
