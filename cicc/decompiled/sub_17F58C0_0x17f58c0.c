// Function: sub_17F58C0
// Address: 0x17f58c0
//
_QWORD *__fastcall sub_17F58C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const char *a5)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // r12
  __int64 v12; // rax
  size_t v13; // rax
  size_t v14; // r8
  _QWORD *v15; // rdx
  int v16; // eax
  size_t v17; // rdx
  const char *v18; // rsi
  unsigned int v19; // esi
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 *n; // [rsp+0h] [rbp-80h]
  size_t na; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  _QWORD v26[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v27[2]; // [rsp+20h] [rbp-60h] BYREF
  const char *v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  _QWORD v30[8]; // [rsp+40h] [rbp-40h] BYREF

  n = sub_1645D80((__int64 *)a4, a2);
  v25 = sub_15A06D0((__int64 **)n, a2, v9, v10);
  v28 = "__sancov_gen_";
  LOWORD(v30[0]) = 259;
  v11 = sub_1648A60(88, 1u);
  if ( v11 )
    sub_15E51E0((__int64)v11, *(_QWORD *)(a1 + 368), (__int64)n, 0, 8, v25, (__int64)&v28, 0, 0, 0, 0);
  v12 = *(_QWORD *)(a3 + 48);
  if ( v12 )
    v11[6] = v12;
  v26[0] = v27;
  v13 = strlen(a5);
  v28 = (const char *)v13;
  v14 = v13;
  if ( v13 > 0xF )
  {
    na = v13;
    v21 = sub_22409D0(v26, &v28, 0);
    v14 = na;
    v26[0] = v21;
    v22 = (_QWORD *)v21;
    v27[0] = v28;
  }
  else
  {
    if ( v13 == 1 )
    {
      LOBYTE(v27[0]) = *a5;
      v15 = v27;
      goto LABEL_8;
    }
    if ( !v13 )
    {
      v15 = v27;
      goto LABEL_8;
    }
    v22 = v27;
  }
  memcpy(v22, a5, v14);
  v13 = (size_t)v28;
  v15 = (_QWORD *)v26[0];
LABEL_8:
  v26[1] = v13;
  *((_BYTE *)v15 + v13) = 0;
  v16 = *(_DWORD *)(a1 + 428);
  if ( v16 == 1 )
  {
    v17 = 7;
    v28 = (const char *)v30;
    v18 = (const char *)v30;
    v30[0] = 0x4D24564F43532ELL;
    v29 = 7;
  }
  else
  {
    if ( v16 == 3 )
      sub_8FD6D0((__int64)&v28, "__DATA,__", v26);
    else
      sub_8FD6D0((__int64)&v28, "__", v26);
    v17 = v29;
    v18 = v28;
  }
  sub_15E5D20((__int64)v11, v18, v17);
  if ( v28 != (const char *)v30 )
    j_j___libc_free_0(v28, v30[0] + 1LL);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0], v27[0] + 1LL);
  if ( *(_BYTE *)(a4 + 8) == 15 )
    v19 = sub_15A9520(*(_QWORD *)(a1 + 440), 0);
  else
    v19 = (unsigned int)sub_1643030(a4) >> 3;
  sub_15E4CC0((__int64)v11, v19);
  return v11;
}
