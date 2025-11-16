// Function: sub_CA1590
// Address: 0xca1590
//
__int64 sub_CA1590()
{
  __int64 v0; // r12
  int v1; // edx
  __int64 *v2; // rbx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  char v6; // al

  v0 = sub_22077B0(200);
  if ( v0 )
  {
    *(_QWORD *)v0 = &unk_49DC150;
    v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    *(_DWORD *)(v0 + 12) &= 0x8000u;
    *(_DWORD *)(v0 + 8) = v1;
    *(_QWORD *)(v0 + 80) = 0x100000000LL;
    *(_WORD *)(v0 + 16) = 0;
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
    v5 = *(unsigned int *)(v0 + 80);
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(v0 + 84) )
    {
      sub_C8D5F0(v0 + 72, (const void *)(v0 + 88), v5 + 1, 8u, v3, v4);
      v5 = *(unsigned int *)(v0 + 80);
    }
    *(_QWORD *)(*(_QWORD *)(v0 + 72) + 8 * v5) = v2;
    *(_WORD *)(v0 + 152) = 0;
    ++*(_DWORD *)(v0 + 80);
    *(_BYTE *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 144) = &unk_49D9748;
    *(_QWORD *)v0 = &unk_49DC090;
    *(_QWORD *)(v0 + 160) = &unk_49DC1D0;
    *(_QWORD *)(v0 + 192) = nullsub_23;
    *(_QWORD *)(v0 + 184) = sub_984030;
    sub_C53080(v0, (__int64)"treat-scalable-fixed-error-as-warning", 37);
    v6 = *(_BYTE *)(v0 + 12);
    *(_QWORD *)(v0 + 48) = 109;
    *(_BYTE *)(v0 + 12) = v6 & 0x9F | 0x20;
    *(_QWORD *)(v0 + 40) = "Treat issues where a fixed-width property is requested from a scalable type as a warning, ins"
                           "tead of an error";
    sub_C53130(v0);
  }
  return v0;
}
