// Function: sub_C8B920
// Address: 0xc8b920
//
__int64 sub_C8B920()
{
  __int64 v0; // r12
  int v1; // edx
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  bool v8; // zf
  __int64 v9; // rax
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+20h] [rbp-30h]
  char v13; // [rsp+21h] [rbp-2Fh]

  v0 = sub_22077B0(200);
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
    *(_QWORD *)(v0 + 144) = &unk_49D9748;
    *(_BYTE *)(v0 + 153) = 0;
    *(_QWORD *)v0 = &unk_49D9AD8;
    *(_QWORD *)(v0 + 160) = &unk_49DC1D0;
    *(_QWORD *)(v0 + 192) = nullsub_39;
    *(_QWORD *)(v0 + 184) = sub_AA4180;
    sub_C53080(v0, (__int64)"disable-symbolication", 21);
    v8 = *(_QWORD *)(v0 + 136) == 0;
    *(_QWORD *)(v0 + 48) = 37;
    *(_QWORD *)(v0 + 40) = "Disable symbolizing crash backtraces.";
    if ( v8 )
    {
      *(_BYTE *)(v0 + 153) = 1;
      *(_QWORD *)(v0 + 136) = &byte_4F84D18;
      *(_BYTE *)(v0 + 152) = byte_4F84D18;
    }
    else
    {
      v9 = sub_CEADF0(v0, "disable-symbolication", v4, v5, v6, v7);
      v13 = 1;
      v11 = "cl::location(x) specified more than once!";
      v12 = 3;
      sub_C53280(v0, (__int64)&v11, 0, 0, v9);
    }
    *(_BYTE *)(v0 + 12) = *(_BYTE *)(v0 + 12) & 0x9F | 0x20;
    sub_C53130(v0);
  }
  return v0;
}
