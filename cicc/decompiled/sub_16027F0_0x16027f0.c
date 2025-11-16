// Function: sub_16027F0
// Address: 0x16027f0
//
__int64 __fastcall sub_16027F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  char v8; // di
  char *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r9
  const char *v12; // r15
  size_t v13; // rax
  _WORD *v14; // rdi
  size_t v15; // rdx
  unsigned __int64 v16; // rax
  void **v17; // r8
  char *v18; // rax
  __int64 v19; // rax
  void **v20; // [rsp+0h] [rbp-A0h]
  size_t v21; // [rsp+8h] [rbp-98h]
  void *v22; // [rsp+10h] [rbp-90h] BYREF
  void **v23; // [rsp+18h] [rbp-88h]
  _QWORD v24[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  void *v26; // [rsp+40h] [rbp-60h] BYREF
  char *v27; // [rsp+48h] [rbp-58h]
  _BYTE *v28; // [rsp+50h] [rbp-50h]
  void *dest; // [rsp+58h] [rbp-48h]
  int v30; // [rsp+60h] [rbp-40h]
  _QWORD *v31; // [rsp+68h] [rbp-38h]

  if ( (unsigned int)(*(_DWORD *)(a2 + 8) - 8) <= 8 )
  {
    v3 = sub_1602790(a1);
    v4 = (__int64 *)v3;
    if ( v3 )
    {
      v26 = (void *)a2;
      nullsub_622(v3);
      if ( (unsigned __int8)sub_16E4B20(v4, 0) )
      {
        (*(void (__fastcall **)(__int64 *))(*v4 + 104))(v4);
        sub_15CAD70(v4, (__int64 *)&v26);
        (*(void (__fastcall **)(__int64 *))(*v4 + 112))(v4);
        nullsub_628(v4);
      }
      sub_16E4BA0(v4);
    }
  }
  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
  if ( v6 )
  {
    if ( *(_BYTE *)(v5 + 96) )
    {
      if ( !(unsigned __int8)sub_1602570(a2) )
        goto LABEL_9;
      v6 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
    }
    result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 16LL))(v6, a2);
    if ( (_BYTE)result )
      return result;
  }
LABEL_9:
  result = sub_1602570(a2);
  if ( !(_BYTE)result )
    return result;
  LOBYTE(v25[0]) = 0;
  v8 = *(_BYTE *)(a2 + 12);
  v24[0] = v25;
  v24[1] = 0;
  v26 = &unk_49EFBE0;
  v30 = 1;
  dest = 0;
  v28 = 0;
  v27 = 0;
  v31 = v24;
  v22 = &unk_49ED080;
  v23 = &v26;
  v9 = sub_16027C0(v8);
  v12 = v9;
  if ( v9 )
  {
    v13 = strlen(v9);
    v14 = dest;
    v15 = v13;
    v16 = v28 - (_BYTE *)dest;
    if ( v15 > v28 - (_BYTE *)dest )
    {
      v19 = sub_16E7EE0(&v26, v12);
      v14 = *(_WORD **)(v19 + 24);
      v17 = (void **)v19;
      if ( *(_QWORD *)(v19 + 16) - (_QWORD)v14 > 1u )
        goto LABEL_16;
LABEL_26:
      sub_16E7EE0(v17, ": ", 2, v10, v17, v11, v20, v21, v22, v23);
      goto LABEL_17;
    }
    v17 = &v26;
    if ( v15 )
    {
      v21 = v15;
      v20 = &v26;
      memcpy(dest, v12, v15);
      dest = (char *)dest + v21;
      v14 = dest;
      v17 = &v26;
      v16 = v28 - (_BYTE *)dest;
    }
  }
  else
  {
    v14 = dest;
    v17 = &v26;
    v16 = v28 - (_BYTE *)dest;
  }
  if ( v16 <= 1 )
    goto LABEL_26;
LABEL_16:
  *v14 = 8250;
  v17[3] = (char *)v17[3] + 2;
LABEL_17:
  (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a2 + 16LL))(a2, &v22);
  if ( v28 == dest )
  {
    sub_16E7EE0(&v26, "\n", 1);
    v18 = (char *)dest;
  }
  else
  {
    *(_BYTE *)dest = 10;
    v18 = (char *)dest + 1;
    dest = (char *)dest + 1;
  }
  if ( v27 != v18 )
    sub_16E7BA0(&v26);
  sub_1C3EFD0(v24, 1);
  result = sub_16E7BC0(&v26);
  if ( (_QWORD *)v24[0] != v25 )
    return j_j___libc_free_0(v24[0], v25[0] + 1LL);
  return result;
}
