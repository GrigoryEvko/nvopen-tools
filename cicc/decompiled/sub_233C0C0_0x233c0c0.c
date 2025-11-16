// Function: sub_233C0C0
// Address: 0x233c0c0
//
__int64 *__fastcall sub_233C0C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v8; // rdx
  const void *v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  size_t v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v18; // eax
  unsigned int v19; // ebx
  __int64 v20; // rdx
  __int64 v21; // r13
  const void *v22; // [rsp+0h] [rbp-B0h] BYREF
  unsigned __int64 v23; // [rsp+8h] [rbp-A8h]
  const void *v24; // [rsp+10h] [rbp-A0h] BYREF
  size_t v25; // [rsp+18h] [rbp-98h]
  unsigned __int64 v26[4]; // [rsp+20h] [rbp-90h] BYREF
  char *v27[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v28[2]; // [rsp+50h] [rbp-60h] BYREF
  char v29; // [rsp+60h] [rbp-50h]
  _QWORD v30[2]; // [rsp+68h] [rbp-48h] BYREF
  _QWORD *v31; // [rsp+78h] [rbp-38h] BYREF

  v22 = (const void *)a4;
  v23 = a5;
  if ( a5 != 7 )
  {
    if ( a5 )
      goto LABEL_10;
    goto LABEL_17;
  }
  if ( *(_DWORD *)a4 == 1634100580 && *(_WORD *)(a4 + 4) == 27765 && *(_BYTE *)(a4 + 6) == 116 )
  {
    sub_23A1380(v27);
    sub_23038C0(a3, v27, v13, v14, v15, v16);
    if ( (_QWORD *)v27[0] != v28 )
      _libc_free((unsigned __int64)v27[0]);
LABEL_17:
    *a1 = 1;
    v27[0] = 0;
    sub_9C66B0((__int64 *)v27);
    return a1;
  }
  while ( 1 )
  {
LABEL_10:
    v24 = 0;
    v25 = 0;
    LOBYTE(v27[0]) = 44;
    v12 = sub_C931B0((__int64 *)&v22, v27, 1u, 0);
    if ( v12 == -1 )
    {
      v9 = v22;
      v12 = v23;
      v10 = 0;
      v11 = 0;
    }
    else
    {
      v8 = v12 + 1;
      v9 = v22;
      if ( v12 + 1 > v23 )
      {
        v8 = v23;
        v10 = 0;
      }
      else
      {
        v10 = v23 - v8;
      }
      v11 = (__int64)v22 + v8;
      if ( v12 > v23 )
        v12 = v23;
    }
    v24 = v9;
    v25 = v12;
    v22 = (const void *)v11;
    v23 = v10;
    if ( !sub_233BD40(a2, a3, v9, v12) )
      break;
    if ( !v23 )
      goto LABEL_17;
  }
  v18 = sub_C63BB0();
  v27[1] = (char *)33;
  v19 = v18;
  v21 = v20;
  v27[0] = "unknown alias analysis name '{0}'";
  v28[0] = &v31;
  v28[1] = 1;
  v29 = 1;
  v30[0] = &unk_49DB108;
  v30[1] = &v24;
  v31 = v30;
  sub_23328D0((__int64)v26, (__int64)v27);
  sub_23058C0(a1, (__int64)v26, v19, v21);
  sub_2240A30(v26);
  return a1;
}
