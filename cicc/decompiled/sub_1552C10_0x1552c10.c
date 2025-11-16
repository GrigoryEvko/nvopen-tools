// Function: sub_1552C10
// Address: 0x1552c10
//
void __fastcall sub_1552C10(__int64 *a1, __int64 a2)
{
  const char *v3; // r12
  char v4; // al
  __int64 v5; // rbx
  __int64 v6; // r8
  char v7; // al
  unsigned __int8 v8; // di
  char v9; // al
  size_t v10; // rdx
  const char *v11; // rsi
  __int64 v12; // rax
  char v13; // al
  const char *v14; // rsi
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdx
  unsigned int v29; // [rsp+Ch] [rbp-84h]
  const char *v30; // [rsp+10h] [rbp-80h] BYREF
  __int64 v31; // [rsp+18h] [rbp-78h]
  _QWORD v32[14]; // [rsp+20h] [rbp-70h] BYREF

  v3 = (const char *)a2;
  if ( (unsigned __int8)sub_15E4B00(a2) )
    sub_1263B40(*a1, "; Materializable\n");
  sub_1550E20(*a1, a2, (__int64)(a1 + 5), a1[4], *(_QWORD *)(a2 + 40));
  sub_1263B40(*a1, " = ");
  v4 = sub_15E4F60(a2);
  LOBYTE(a2) = *(_BYTE *)(a2 + 32);
  if ( v4 )
  {
    LODWORD(a2) = a2 & 0xF;
    if ( (_DWORD)a2 )
      goto LABEL_5;
    sub_1263B40(*a1, "external ");
    LODWORD(a2) = *((unsigned __int8 *)v3 + 32);
  }
  LOBYTE(a2) = a2 & 0xF;
LABEL_5:
  v5 = *a1;
  sub_1549EC0((__int64)&v30, (unsigned __int8)a2);
  sub_16E7EE0(v5, v30, v31);
  if ( v30 != (const char *)v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  sub_154A4E0((__int64)v3, *a1);
  v6 = *a1;
  v7 = ((unsigned __int8)v3[32] >> 4) & 3;
  if ( v7 == 1 )
  {
    sub_1263B40(*a1, "hidden ");
    v6 = *a1;
  }
  else if ( v7 == 2 )
  {
    sub_1263B40(*a1, "protected ");
    v6 = *a1;
  }
  v8 = v3[33];
  if ( (v8 & 3) == 1 )
  {
    sub_1263B40(v6, "dllimport ");
    v6 = *a1;
    v8 = v3[33];
  }
  else if ( (v8 & 3) == 2 )
  {
    sub_1263B40(v6, "dllexport ");
    v6 = *a1;
    v8 = v3[33];
  }
  sub_154A050((v8 >> 2) & 7, v6);
  v9 = (unsigned __int8)v3[32] >> 6;
  if ( v9 == 1 )
  {
    v10 = 18;
    v11 = "local_unnamed_addr";
  }
  else
  {
    v10 = 12;
    v11 = "unnamed_addr";
    if ( v9 != 2 )
      goto LABEL_16;
  }
  v12 = sub_1549FF0(*a1, v11, v10);
  sub_1549FC0(v12, 0x20u);
LABEL_16:
  if ( *(_DWORD *)(*(_QWORD *)v3 + 8LL) >> 8 )
  {
    v29 = *(_DWORD *)(*(_QWORD *)v3 + 8LL) >> 8;
    v20 = sub_1263B40(*a1, "addrspace(");
    v21 = sub_16E7A90(v20, v29);
    sub_1263B40(v21, ") ");
    v13 = v3[80];
    if ( (v13 & 2) == 0 )
      goto LABEL_18;
  }
  else
  {
    v13 = v3[80];
    if ( (v13 & 2) == 0 )
      goto LABEL_18;
  }
  sub_1263B40(*a1, "externally_initialized ");
  v13 = v3[80];
LABEL_18:
  v14 = "constant ";
  if ( (v13 & 1) == 0 )
    v14 = "global ";
  sub_1263B40(*a1, v14);
  sub_154DAA0((__int64)(a1 + 5), *((_QWORD *)v3 + 3), *a1);
  if ( !(unsigned __int8)sub_15E4F60(v3) )
  {
    sub_1549FC0(*a1, 0x20u);
    sub_15520E0(a1, *((__int64 **)v3 - 3), 0);
  }
  if ( (v3[34] & 0x20) != 0 )
  {
    sub_1263B40(*a1, ", section \"");
    v25 = *a1;
    v26 = 0;
    v27 = 0;
    if ( (v3[34] & 0x20) != 0 )
    {
      v27 = sub_15E61A0(v3, 0, v23, v24);
      v26 = v28;
    }
    sub_16D16F0(v27, v26, v25);
    sub_1549FC0(*a1, 0x22u);
  }
  sub_154B830(*a1, (__int64)v3);
  if ( (unsigned int)(1 << (*((_DWORD *)v3 + 8) >> 15)) >> 1 )
  {
    v22 = sub_1263B40(*a1, ", align ");
    sub_16E7A90(v22, (unsigned int)(1 << (*((_DWORD *)v3 + 8) >> 15)) >> 1);
  }
  v30 = (const char *)v32;
  v31 = 0x400000000LL;
  sub_1626D60(v3, &v30);
  sub_1550BA0(a1, (unsigned int *)&v30, ", ", 2u);
  v15 = *((_QWORD *)v3 + 9);
  if ( v15 )
  {
    v16 = sub_1263B40(*a1, " #");
    v19 = sub_154F2E0(a1[4], v15, v17, v18);
    sub_16E7AB0(v16, v19);
  }
  sub_1552170(a1, v3);
  if ( v30 != (const char *)v32 )
    _libc_free((unsigned __int64)v30);
}
