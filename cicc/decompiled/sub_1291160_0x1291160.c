// Function: sub_1291160
// Address: 0x1291160
//
__int64 __fastcall sub_1291160(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  const char *v6; // rdi
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  _BYTE *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r14
  _BYTE *v22; // [rsp-90h] [rbp-90h]
  __int64 v23; // [rsp-90h] [rbp-90h]
  int v24; // [rsp-7Ch] [rbp-7Ch] BYREF
  _BYTE *v25; // [rsp-78h] [rbp-78h] BYREF
  __int64 v26; // [rsp-70h] [rbp-70h]
  _BYTE v27[16]; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v28; // [rsp-58h] [rbp-58h]
  __int64 v29; // [rsp-50h] [rbp-50h]
  _QWORD v30[9]; // [rsp-48h] [rbp-48h] BYREF

  result = dword_4D046B4;
  if ( dword_4D046B4 )
    return result;
  if ( !a3 )
    return result;
  v6 = *(const char **)(a3 + 64);
  if ( !v6 )
    return result;
  if ( sscanf(v6, "unroll %d", &v24) != 1 )
    sub_127B550("Parsing unroll count failed!", (_DWORD *)a3, 1);
  if ( v24 <= 0 )
    sub_127B550("Unroll count must be positive.", (_DWORD *)a3, 1);
  v8 = *(_QWORD *)(a1 + 40);
  v25 = v27;
  v26 = 0x200000000LL;
  if ( v24 != 0x7FFFFFFF )
  {
    v9 = sub_161FF10(v8, "llvm.loop.unroll.count", 22);
    v22 = (_BYTE *)v24;
    v10 = sub_1643350(*(_QWORD *)(a1 + 40));
    v11 = v22;
    v13 = sub_159C470(v10, v22, 0);
    v14 = (unsigned int)v26;
    if ( (unsigned int)v26 >= HIDWORD(v26) )
    {
      v11 = v27;
      v23 = v13;
      sub_16CD150(&v25, v27, 0, 8);
      v14 = (unsigned int)v26;
      v13 = v23;
    }
    v15 = v25;
    *(_QWORD *)&v25[8 * v14] = v9;
    LODWORD(v26) = v26 + 1;
    v16 = sub_1624210(v13, v11, v15, v12);
    v17 = (unsigned int)v26;
    if ( (unsigned int)v26 < HIDWORD(v26) )
      goto LABEL_10;
    goto LABEL_17;
  }
  v16 = sub_161FF10(v8, "llvm.loop.unroll.full", 21);
  v17 = (unsigned int)v26;
  if ( (unsigned int)v26 >= HIDWORD(v26) )
  {
LABEL_17:
    sub_16CD150(&v25, v27, 0, 8);
    v17 = (unsigned int)v26;
  }
LABEL_10:
  *(_QWORD *)&v25[8 * v17] = v16;
  v18 = *(_QWORD *)(a1 + 40);
  LODWORD(v26) = v26 + 1;
  v19 = sub_1627350(v18, v25, (unsigned int)v26, 0, 1);
  v20 = *(_QWORD *)(a1 + 40);
  v30[1] = v19;
  v28 = v30;
  v29 = 0x200000002LL;
  v30[0] = 0;
  v21 = sub_1627350(v20, v30, 2, 0, 1);
  sub_1630830(v21, 0, v21);
  result = sub_1626100(a2, "llvm.loop", 9, v21);
  if ( v28 != v30 )
    result = _libc_free(v28, "llvm.loop");
  if ( v25 != v27 )
    return _libc_free(v25, "llvm.loop");
  return result;
}
