// Function: sub_14A0550
// Address: 0x14a0550
//
void __fastcall sub_14A0550(__int64 a1)
{
  _QWORD v1[2]; // [rsp-50h] [rbp-58h] BYREF
  _QWORD v2[9]; // [rsp-40h] [rbp-48h] BYREF

  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 96) = -1;
  *(_WORD *)(a1 + 104) = -1;
  *(_OWORD *)a1 = -1;
  *(_OWORD *)(a1 + 16) = -1;
  *(_OWORD *)(a1 + 32) = -1;
  *(_OWORD *)(a1 + 48) = -1;
  *(_OWORD *)(a1 + 64) = -1;
  *(_OWORD *)(a1 + 80) = -1;
  v1[0] = v2;
  v1[1] = 0;
  LOBYTE(v2[0]) = 0;
  memset(&v2[2], 0, 24);
  sub_149FA60(a1, v1);
  if ( (_QWORD *)v1[0] != v2 )
    j_j___libc_free_0(v1[0], v2[0] + 1LL);
}
