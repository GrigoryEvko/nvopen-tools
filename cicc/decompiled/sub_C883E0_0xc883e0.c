// Function: sub_C883E0
// Address: 0xc883e0
//
__int64 sub_C883E0()
{
  __int64 v0; // r12
  int v1; // edx
  __int64 *v2; // rbx
  __int64 v3; // rax
  char v4; // al

  v0 = sub_22077B0(208);
  if ( v0 )
  {
    *(_QWORD *)v0 = &unk_49DC150;
    v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    *(_DWORD *)(v0 + 12) &= 0x8000u;
    *(_WORD *)(v0 + 16) = 0;
    *(_QWORD *)(v0 + 80) = 0x100000000LL;
    *(_DWORD *)(v0 + 8) = v1;
    *(_QWORD *)(v0 + 24) = 0;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 56) = 0;
    *(_QWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = v0 + 88;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = v0 + 128;
    *(_QWORD *)(v0 + 112) = 1;
    *(_DWORD *)(v0 + 120) = 0;
    *(_BYTE *)(v0 + 124) = 1;
    v2 = sub_C57470();
    v3 = *(unsigned int *)(v0 + 80);
    if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(v0 + 84) )
    {
      sub_C8D5F0(v0 + 72, v0 + 88, v3 + 1, 8);
      v3 = *(unsigned int *)(v0 + 80);
    }
    *(_QWORD *)(*(_QWORD *)(v0 + 72) + 8 * v3) = v2;
    ++*(_DWORD *)(v0 + 80);
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 144) = &unk_49DB998;
    *(_QWORD *)(v0 + 152) = 0;
    *(_BYTE *)(v0 + 160) = 0;
    *(_QWORD *)v0 = &unk_49DB9B8;
    *(_QWORD *)(v0 + 168) = &unk_49DC2C0;
    *(_QWORD *)(v0 + 200) = nullsub_121;
    *(_QWORD *)(v0 + 192) = sub_C1A370;
    sub_C53080(v0, (__int64)"rng-seed", 8);
    *(_QWORD *)(v0 + 64) = 4;
    *(_QWORD *)(v0 + 56) = "seed";
    v4 = *(_BYTE *)(v0 + 12);
    *(_QWORD *)(v0 + 48) = 36;
    *(_QWORD *)(v0 + 136) = 0;
    *(_BYTE *)(v0 + 160) = 1;
    *(_BYTE *)(v0 + 12) = v4 & 0x9F | 0x20;
    *(_QWORD *)(v0 + 40) = "Seed for the random number generator";
    *(_QWORD *)(v0 + 152) = 0;
    sub_C53130(v0);
  }
  return v0;
}
