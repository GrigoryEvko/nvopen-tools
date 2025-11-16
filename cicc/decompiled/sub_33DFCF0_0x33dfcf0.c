// Function: sub_33DFCF0
// Address: 0x33dfcf0
//
bool __fastcall sub_33DFCF0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r13
  unsigned int v8; // edx
  __int64 v9; // rsi
  unsigned __int16 *v10; // rdx
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v23; // rdx
  __int16 v24; // [rsp-68h] [rbp-68h] BYREF
  __int64 v25; // [rsp-60h] [rbp-60h]
  __int16 v26; // [rsp-58h] [rbp-58h] BYREF
  __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 24) != 188 )
    return 0;
  v7 = sub_33CF5B0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL));
  v9 = v8;
  v10 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * v8);
  v11 = *((_QWORD *)v10 + 1);
  v12 = *v10;
  v24 = v12;
  v25 = v11;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      v26 = v12;
      v27 = v11;
      goto LABEL_14;
    }
    LOWORD(v12) = word_4456580[v12 - 1];
    v23 = 0;
LABEL_20:
    v26 = v12;
    v27 = v23;
    if ( !(_WORD)v12 )
      goto LABEL_6;
LABEL_14:
    if ( (_WORD)v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      BUG();
    v16 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v12 - 16];
    goto LABEL_7;
  }
  if ( sub_30070B0((__int64)&v24) )
  {
    LOWORD(v12) = sub_3009970((__int64)&v24, v9, v13, v14, v15);
    goto LABEL_20;
  }
  v27 = v11;
  v26 = 0;
LABEL_6:
  v28 = sub_3007260((__int64)&v26);
  LODWORD(v16) = v28;
  v29 = v17;
LABEL_7:
  v18 = sub_33DFBC0(v7, v9, a3, 1u, v5, v6);
  if ( !v18 )
    return 0;
  v19 = *(_QWORD *)(v18 + 96);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
  {
    v20 = sub_C445E0(v19 + 24);
  }
  else
  {
    v20 = 64;
    _RDX = ~*(_QWORD *)(v19 + 24);
    __asm { tzcnt   rcx, rdx }
    if ( *(_QWORD *)(v19 + 24) != -1 )
      v20 = _RCX;
  }
  return v20 >= (unsigned int)v16;
}
