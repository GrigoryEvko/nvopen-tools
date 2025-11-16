// Function: sub_1635D20
// Address: 0x1635d20
//
__int64 __fastcall sub_1635D20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v7; // rcx
  __int64 **v8; // rbx
  __int64 **v9; // r15
  __int64 *v10; // rcx
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  const char *v13; // rsi
  const char *v14; // r12
  size_t v15; // rbx
  const char *v16; // rax
  size_t v17; // rdx
  __int64 *v19; // [rsp+18h] [rbp-58h]
  const char *v20; // [rsp+20h] [rbp-50h] BYREF
  size_t v21; // [rsp+28h] [rbp-48h]
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = 1;
  if ( !*(_BYTE *)(a1 + 8) )
    return v3;
  v20 = (const char *)v22;
  sub_1634F50((__int64 *)&v20, "SCC (", (__int64)"");
  v8 = *(__int64 ***)(a3 + 16);
  v9 = *(__int64 ***)(a3 + 24);
  if ( v8 != v9 )
  {
    v10 = *v8;
    v11 = **v8;
    if ( !v11 )
      goto LABEL_9;
LABEL_5:
    v13 = (const char *)sub_1649960(v11);
    if ( v12 > 0x3FFFFFFFFFFFFFFFLL - v21 )
LABEL_14:
      sub_4262D8((__int64)"basic_string::append");
    while ( 1 )
    {
      ++v8;
      sub_2241490(&v20, v13, v12, v10);
      if ( v9 == v8 )
        break;
      if ( 0x3FFFFFFFFFFFFFFFLL - v21 <= 1 )
        goto LABEL_14;
      v19 = *v8;
      sub_2241490(&v20, ", ", 2, *v8);
      v10 = v19;
      v11 = *v19;
      if ( *v19 )
        goto LABEL_5;
LABEL_9:
      if ( 0x3FFFFFFFFFFFFFFFLL - v21 <= 0x10 )
        goto LABEL_14;
      v12 = 17;
      v13 = "<<null function>>";
    }
  }
  if ( v21 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_14;
  sub_2241490(&v20, ")", 1, v7);
  v14 = v20;
  v15 = v21;
  v16 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
  v3 = sub_1635030(a1, v16, v17, v14, v15);
  if ( v20 != (const char *)v22 )
    j_j___libc_free_0(v20, v22[0] + 1LL);
  return v3;
}
