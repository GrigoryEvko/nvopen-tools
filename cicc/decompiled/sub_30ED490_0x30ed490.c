// Function: sub_30ED490
// Address: 0x30ed490
//
__int64 __fastcall sub_30ED490(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  char *v4; // rax
  size_t v5; // rdx
  __int8 *v6; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  __int8 *v10; // rax
  size_t v11; // rdx
  __int8 *v12; // [rsp+0h] [rbp-90h] BYREF
  size_t v13; // [rsp+8h] [rbp-88h]
  __int64 v14; // [rsp+10h] [rbp-80h]
  _BYTE v15[24]; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v16[12]; // [rsp+30h] [rbp-60h] BYREF

  sub_B174A0(a1, (__int64)"kernel-info", (__int64)"FlatAddrspaceAccess", 19, *a2);
  sub_B18290(a1, "in ", 3u);
  sub_30ED1B0(a1, *(_QWORD *)(a2[1] + 40), (unsigned __int8 *)a2[1], "function", 8u);
  v3 = *a2;
  if ( *(_BYTE *)*a2 == 85
    && (v8 = *(_QWORD *)(v3 - 32)) != 0
    && !*(_BYTE *)v8
    && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v3 + 80)
    && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
  {
    sub_B18290(a1, ", '", 3u);
    v9 = *(_QWORD *)(v3 - 32);
    if ( v9 )
    {
      if ( *(_BYTE *)v9 )
      {
        v9 = 0;
      }
      else if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(v3 + 80) )
      {
        v9 = 0;
      }
    }
    v10 = (__int8 *)sub_BD5D20(v9);
    sub_B18290(a1, v10, v11);
    sub_B18290(a1, "' call", 6u);
  }
  else
  {
    sub_B18290(a1, ", '", 3u);
    v4 = sub_B458E0((unsigned int)*(unsigned __int8 *)*a2 - 29);
    v5 = 0;
    v6 = v4;
    if ( v4 )
      v5 = strlen(v4);
    sub_B18290(a1, v6, v5);
    sub_B18290(a1, "' instruction", 0xDu);
  }
  if ( *(_BYTE *)(*(_QWORD *)(*a2 + 8) + 8LL) != 7 )
  {
    v16[5] = 0x100000000LL;
    v16[6] = &v12;
    v16[0] = &unk_49DD288;
    v12 = v15;
    v13 = 0;
    v14 = 20;
    v16[1] = 2;
    memset(&v16[2], 0, 24);
    sub_CB5980((__int64)v16, 0, 0, 0);
    sub_A5BF40((unsigned __int8 *)*a2, (__int64)v16, 0, *(_QWORD *)(a2[1] + 40));
    sub_B18290(a1, " ('", 3u);
    sub_B18290(a1, v12, v13);
    sub_B18290(a1, "')", 2u);
    v16[0] = &unk_49DD388;
    sub_CB5840((__int64)v16);
    if ( v12 != v15 )
      _libc_free((unsigned __int64)v12);
  }
  sub_B18290(a1, " accesses memory in flat address space", 0x26u);
  return a1;
}
