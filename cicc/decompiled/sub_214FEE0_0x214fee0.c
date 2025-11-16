// Function: sub_214FEE0
// Address: 0x214fee0
//
__int64 __fastcall sub_214FEE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  _BYTE *v9; // rax
  _QWORD *v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  int v13; // eax
  __int64 v14; // rdi
  _WORD *v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v19; // r14
  __int64 v20; // r14
  __int64 v21; // rax
  const char *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  char *v25[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF

  v6 = sub_396DDB0();
  v7 = *(_QWORD *)(a2 + 24);
  v8 = v6;
  v9 = *(_BYTE **)(a3 + 24);
  if ( *(_BYTE **)(a3 + 16) == v9 )
  {
    sub_16E7EE0(a3, ".", 1u);
  }
  else
  {
    *v9 = 46;
    ++*(_QWORD *)(a3 + 24);
  }
  sub_214FA80(a1, *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8, a3);
  if ( (unsigned __int8)sub_1C2E7D0(a2) )
    sub_1263B40(a3, " .attribute(.managed)");
  v10 = *(_QWORD **)(a3 + 24);
  v11 = *(_QWORD *)(a3 + 16) - (_QWORD)v10;
  if ( (unsigned int)(1 << (*(_DWORD *)(a2 + 32) >> 15)) >> 1 )
  {
    if ( v11 <= 7 )
    {
      v14 = sub_16E7EE0(a3, " .align ", 8u);
    }
    else
    {
      v14 = a3;
      *v10 = 0x206E67696C612E20LL;
      *(_QWORD *)(a3 + 24) += 8LL;
    }
    sub_16E7A90(v14, (unsigned int)(1 << (*(_DWORD *)(a2 + 32) >> 15)) >> 1);
  }
  else
  {
    if ( v11 <= 7 )
    {
      v12 = sub_16E7EE0(a3, " .align ", 8u);
    }
    else
    {
      v12 = a3;
      *v10 = 0x206E67696C612E20LL;
      *(_QWORD *)(a3 + 24) += 8LL;
    }
    v24 = v12;
    v13 = sub_15AAE50(v8, v7);
    sub_16E7AB0(v24, v13);
  }
  if ( sub_1642F90(v7, 128) )
  {
    sub_1263B40(a3, " .b8 ");
    v23 = sub_396EAF0(a1, a2);
    sub_38E2490(v23, a3, *(_QWORD *)(a1 + 240));
    v22 = "[16]";
    return sub_1263B40(a3, v22);
  }
  if ( (*(_BYTE *)(v7 + 8) & 0xFB) != 0xB && (unsigned __int8)(*(_BYTE *)(v7 + 8) - 1) > 5u )
  {
    v19 = sub_127FA20(v8, v7) + 7;
    sub_1263B40(a3, " .b8 ");
    v20 = v19 >> 3;
    v21 = sub_396EAF0(a1, a2);
    sub_38E2490(v21, a3, *(_QWORD *)(a1 + 240));
    sub_1263B40(a3, "[");
    if ( v20 )
      sub_16E7AB0(a3, v20);
    v22 = "]";
    return sub_1263B40(a3, v22);
  }
  v15 = *(_WORD **)(a3 + 24);
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v15 <= 1u )
  {
    sub_16E7EE0(a3, " .", 2u);
  }
  else
  {
    *v15 = 11808;
    *(_QWORD *)(a3 + 24) += 2LL;
  }
  sub_214FBF0((__int64)v25, a1, v7, 1);
  sub_16E7EE0(a3, v25[0], (size_t)v25[1]);
  if ( (__int64 *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0], v26 + 1);
  v16 = *(_BYTE **)(a3 + 24);
  if ( *(_BYTE **)(a3 + 16) == v16 )
  {
    sub_16E7EE0(a3, " ", 1u);
  }
  else
  {
    *v16 = 32;
    ++*(_QWORD *)(a3 + 24);
  }
  v17 = sub_396EAF0(a1, a2);
  return sub_38E2490(v17, a3, *(_QWORD *)(a1 + 240));
}
