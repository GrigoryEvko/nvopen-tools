// Function: sub_17D7CF0
// Address: 0x17d7cf0
//
unsigned __int64 __fastcall sub_17D7CF0(__int128 a1)
{
  __int64 v1; // r12
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 result; // rax
  __int64 v16; // rax
  __int64 v17[16]; // [rsp+0h] [rbp-80h] BYREF

  v1 = *((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)v17, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v2 = *(__int64 **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v2 = (__int64 *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  v3 = *v2;
  v4 = sub_17CFB40(a1, *v2, v17, **((__int64 ***)&a1 + 1), 1u);
  if ( byte_4FA4600 )
  {
    *((_QWORD *)&a1 + 1) = v3;
    sub_17D5820(a1, v1);
    if ( *(_BYTE *)(v1 + 16) != 58 )
      goto LABEL_5;
  }
  else if ( *(_BYTE *)(*((_QWORD *)&a1 + 1) + 16LL) != 58 )
  {
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v16 = *(_QWORD *)(v1 - 8);
  else
    v16 = v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v16 + 24);
  sub_17D5820(a1, v1);
LABEL_5:
  v5 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v1);
  *((_QWORD *)&a1 + 1) = v5;
  if ( v5 )
    *((_QWORD *)&a1 + 1) = sub_15A06D0((__int64 **)v5, (__int64)v5, v6, v7);
  sub_12A8F50(v17, *((__int64 *)&a1 + 1), v4, 0);
  v8 = *(_QWORD *)v1;
  v9 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v1);
  v11 = (__int64)v9;
  if ( v9 )
    v11 = sub_15A06D0((__int64 **)v9, v8, (__int64)v9, v10);
  sub_17D4920(a1, (__int64 *)v1, v11);
  v14 = sub_15A06D0(*(__int64 ***)(*(_QWORD *)(a1 + 8) + 184LL), v1, v12, v13);
  result = sub_17D4B80(a1, v1, v14);
  if ( v17[0] )
    return sub_161E7C0((__int64)v17, v17[0]);
  return result;
}
