// Function: sub_C8B5D0
// Address: 0xc8b5d0
//
__int64 sub_C8B5D0()
{
  __int64 v0; // r13
  __int64 v1; // r12
  int v2; // edx
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  bool v9; // zf
  __int64 v10; // rax
  const char *v12; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  if ( !qword_4F84D00 )
    sub_C7D570(&qword_4F84D00, sub_C8B3D0, (__int64)sub_C8BB60);
  v0 = qword_4F84D00;
  v1 = sub_22077B0(232);
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
    v4 = *(unsigned int *)(v1 + 80);
    if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 84) )
    {
      sub_C8D5F0(v1 + 72, v1 + 88, v4 + 1, 8);
      v4 = *(unsigned int *)(v1 + 80);
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 72) + 8 * v4) = v3;
    *(_QWORD *)(v1 + 152) = v1 + 168;
    ++*(_DWORD *)(v1 + 80);
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 144) = &unk_49DC130;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)v1 = &unk_49DCA98;
    *(_BYTE *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 192) = &unk_49DC350;
    *(_QWORD *)(v1 + 224) = nullsub_164;
    *(_QWORD *)(v1 + 216) = sub_C8B390;
    sub_C53080(v1, (__int64)"crash-diagnostics-dir", 21);
    v9 = *(_QWORD *)(v1 + 136) == 0;
    *(_QWORD *)(v1 + 64) = 9;
    *(_QWORD *)(v1 + 56) = "directory";
    *(_QWORD *)(v1 + 40) = "Directory for crash diagnostic files.";
    *(_QWORD *)(v1 + 48) = 37;
    if ( v9 )
    {
      *(_QWORD *)(v1 + 136) = v0;
      *(_BYTE *)(v1 + 184) = 1;
      sub_2240AE0(v1 + 152, v0);
    }
    else
    {
      v10 = sub_CEADF0(v1, "crash-diagnostics-dir", v5, v6, v7, v8);
      v14 = 1;
      v12 = "cl::location(x) specified more than once!";
      v13 = 3;
      sub_C53280(v1, (__int64)&v12, 0, 0, v10);
    }
    *(_BYTE *)(v1 + 12) = *(_BYTE *)(v1 + 12) & 0x9F | 0x20;
    sub_C53130(v1);
  }
  return v1;
}
