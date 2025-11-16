// Function: sub_A5DDD0
// Address: 0xa5ddd0
//
__int64 __fastcall sub_A5DDD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // ecx
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r15d
  __int64 v15; // rdi
  void *v16; // rdx
  const char *v17; // rax
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  unsigned __int8 v20; // al
  __int64 v21; // rdx
  unsigned __int8 v22; // al
  __int64 v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  unsigned __int8 v26; // al
  __int64 v27; // rbx
  unsigned int v28; // r15d
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  const char *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v38; // [rsp+8h] [rbp-58h]
  __int64 v39; // [rsp+10h] [rbp-50h] BYREF
  char v40; // [rsp+18h] [rbp-48h]
  const char *v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+28h] [rbp-38h]

  sub_904010(a1, "!DICompileUnit(");
  v5 = *(_DWORD *)(a2 + 16);
  v42 = a3;
  v6 = a2 - 16;
  v39 = a1;
  v40 = 1;
  v41 = ", ";
  sub_A53AC0(&v39, "language", 8u, v5, (__int64 (__fastcall *)(_QWORD))sub_E0A700, 0);
  v7 = a2;
  if ( *(_BYTE *)a2 != 16 )
    v7 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v39, "file", 4u, v7, 0);
  v8 = sub_A547D0(a2, 1);
  sub_A53660(&v39, "producer", 8u, v8, v9, 1);
  sub_A53370((__int64)&v39, "isOptimized", 0xBu, *(_BYTE *)(a2 + 40), 0);
  v10 = sub_A547D0(a2, 2);
  sub_A53660(&v39, "flags", 5u, v10, v11, 1);
  sub_A537C0((__int64)&v39, "runtimeVersion", 0xEu, *(_DWORD *)(a2 + 20), 0);
  v12 = sub_A547D0(a2, 3);
  sub_A53660(&v39, "splitDebugFilename", 0x12u, v12, v13, 1);
  v14 = *(_DWORD *)(a2 + 32);
  v15 = v39;
  if ( v40 )
    v40 = 0;
  else
    v15 = sub_904010(v39, v41);
  v16 = *(void **)(v15 + 32);
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0xBu )
  {
    v15 = sub_CB6200(v15, "emissionKind", 12);
  }
  else
  {
    qmemcpy(v16, "emissionKind", 12);
    *(_QWORD *)(v15 + 32) += 12LL;
  }
  v38 = sub_904010(v15, ": ");
  v17 = (const char *)sub_AF33B0(v14);
  sub_904010(v38, v17);
  v18 = *(_BYTE *)(a2 - 16);
  if ( (v18 & 2) != 0 )
    v19 = *(_QWORD *)(a2 - 32);
  else
    v19 = v6 - 8LL * ((v18 >> 2) & 0xF);
  sub_A5CC00((__int64)&v39, "enums", 5u, *(_QWORD *)(v19 + 32), 1);
  v20 = *(_BYTE *)(a2 - 16);
  if ( (v20 & 2) != 0 )
    v21 = *(_QWORD *)(a2 - 32);
  else
    v21 = v6 - 8LL * ((v20 >> 2) & 0xF);
  sub_A5CC00((__int64)&v39, "retainedTypes", 0xDu, *(_QWORD *)(v21 + 40), 1);
  v22 = *(_BYTE *)(a2 - 16);
  if ( (v22 & 2) != 0 )
    v23 = *(_QWORD *)(a2 - 32);
  else
    v23 = v6 - 8LL * ((v22 >> 2) & 0xF);
  sub_A5CC00((__int64)&v39, "globals", 7u, *(_QWORD *)(v23 + 48), 1);
  v24 = *(_BYTE *)(a2 - 16);
  if ( (v24 & 2) != 0 )
    v25 = *(_QWORD *)(a2 - 32);
  else
    v25 = v6 - 8LL * ((v24 >> 2) & 0xF);
  sub_A5CC00((__int64)&v39, "imports", 7u, *(_QWORD *)(v25 + 56), 1);
  v26 = *(_BYTE *)(a2 - 16);
  if ( (v26 & 2) != 0 )
    v27 = *(_QWORD *)(a2 - 32);
  else
    v27 = v6 - 8LL * ((v26 >> 2) & 0xF);
  sub_A5CC00((__int64)&v39, "macros", 6u, *(_QWORD *)(v27 + 64), 1);
  sub_A539C0((__int64)&v39, "dwoId", 5u, *(_QWORD *)(a2 + 24));
  sub_A53370((__int64)&v39, "splitDebugInlining", 0x12u, *(_BYTE *)(a2 + 41), 0x101u);
  sub_A53370((__int64)&v39, "debugInfoForProfiling", 0x15u, *(_BYTE *)(a2 + 42), 0x100u);
  v28 = *(_DWORD *)(a2 + 36);
  if ( v28 )
  {
    v29 = v39;
    if ( v40 )
      v40 = 0;
    else
      v29 = sub_904010(v39, v41);
    v30 = sub_A51340(v29, "nameTableKind", 0xDu);
    v31 = sub_904010(v30, ": ");
    v32 = (const char *)sub_AF33F0(v28);
    sub_904010(v31, v32);
  }
  sub_A53370((__int64)&v39, "rangesBaseAddress", 0x11u, *(_BYTE *)(a2 + 43), 0x100u);
  v33 = sub_A547D0(a2, 9);
  sub_A53660(&v39, "sysroot", 7u, v33, v34, 1);
  v35 = sub_A547D0(a2, 10);
  sub_A53660(&v39, "sdk", 3u, v35, v36, 1);
  return sub_904010(a1, ")");
}
