// Function: sub_CA5520
// Address: 0xca5520
//
__int64 sub_CA5520()
{
  __int64 *v0; // r13
  __int64 v1; // r12
  int v2; // edx
  __int64 *v3; // rbx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax

  v0 = sub_CA5490();
  v1 = sub_22077B0(200);
  if ( v1 )
  {
    *(_QWORD *)v1 = &unk_49DC150;
    v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    *(_DWORD *)(v1 + 12) &= 0x8000u;
    *(_WORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 80) = 0x100000000LL;
    *(_DWORD *)(v1 + 8) = v2;
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 56) = 0;
    *(_QWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = v1 + 88;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = v1 + 128;
    *(_QWORD *)(v1 + 112) = 1;
    *(_DWORD *)(v1 + 120) = 0;
    *(_BYTE *)(v1 + 124) = 1;
    v3 = sub_C57470();
    v6 = *(unsigned int *)(v1 + 80);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 84) )
    {
      sub_C8D5F0(v1 + 72, (const void *)(v1 + 88), v6 + 1, 8u, v4, v5);
      v6 = *(unsigned int *)(v1 + 80);
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 72) + 8 * v6) = v3;
    ++*(_DWORD *)(v1 + 80);
    *(_DWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 144) = &unk_49DC110;
    *(_DWORD *)(v1 + 152) = 0;
    *(_BYTE *)(v1 + 156) = 0;
    *(_QWORD *)v1 = &unk_49D97F0;
    *(_QWORD *)(v1 + 160) = &unk_49DC200;
    *(_QWORD *)(v1 + 192) = nullsub_26;
    *(_QWORD *)(v1 + 184) = sub_9C26D0;
    sub_C53080(v1, (__int64)"color", 5);
    sub_C57500(v1, v0);
    *(_QWORD *)(v1 + 48) = 41;
    *(_QWORD *)(v1 + 40) = "Use colors in output (default=autodetect)";
    *(_DWORD *)(v1 + 136) = 0;
    *(_BYTE *)(v1 + 156) = 1;
    *(_DWORD *)(v1 + 152) = 0;
    sub_C53130(v1);
  }
  return v1;
}
