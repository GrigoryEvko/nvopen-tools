// Function: sub_39D16D0
// Address: 0x39d16d0
//
void __fastcall sub_39D16D0(__int64 a1, __int64 a2)
{
  char *v3; // rdx
  __int64 v4; // rax
  void (__fastcall *v5)(__int64, char **, _QWORD); // rbx
  unsigned int v6; // eax
  __int64 v7; // rax
  void (__fastcall *v8)(__int64, _BYTE **, _QWORD); // r13
  unsigned int v9; // eax
  __int64 v10; // r13
  char *v11; // rdi
  char *v12; // rax
  size_t v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  size_t v19; // rdx
  char *v20; // [rsp+0h] [rbp-80h] BYREF
  char *v21; // [rsp+8h] [rbp-78h]
  _BYTE *v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  _BYTE v24[16]; // [rsp+20h] [rbp-60h] BYREF
  char *p_src; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  __int64 src; // [rsp+40h] [rbp-40h] BYREF
  __int64 v28; // [rsp+48h] [rbp-38h]
  int v29; // [rsp+50h] [rbp-30h]
  char **v30; // [rsp+58h] [rbp-28h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v24[0] = 0;
    v22 = v24;
    v23 = 0;
    v29 = 1;
    p_src = (char *)&unk_49EFBE0;
    v28 = 0;
    src = 0;
    n = 0;
    v30 = &v22;
    sub_16E4080(a1);
    sub_16E7EE0((__int64)&p_src, *(char **)a2, *(_QWORD *)(a2 + 8));
    if ( v28 != n )
      sub_16E7BA0((__int64 *)&p_src);
    v3 = *v30;
    v21 = v30[1];
    v4 = *(_QWORD *)a1;
    v20 = v3;
    v5 = *(void (__fastcall **)(__int64, char **, _QWORD))(v4 + 216);
    v6 = sub_15C8A80(v3, (unsigned __int64)v21);
    v5(a1, &v20, v6);
    sub_16E7BC0((__int64 *)&p_src);
    if ( v22 != v24 )
      j_j___libc_free_0((unsigned __int64)v22);
    return;
  }
  v7 = *(_QWORD *)a1;
  v22 = 0;
  v23 = 0;
  v8 = *(void (__fastcall **)(__int64, _BYTE **, _QWORD))(v7 + 216);
  v9 = sub_15C8A80(0, 0);
  v8(a1, &v22, v9);
  v10 = sub_16E4080(a1);
  if ( !v22 )
  {
    LOBYTE(src) = 0;
    v11 = *(char **)a2;
    v19 = 0;
    p_src = (char *)&src;
LABEL_17:
    *(_QWORD *)(a2 + 8) = v19;
    v11[v19] = 0;
    v12 = p_src;
    goto LABEL_12;
  }
  p_src = (char *)&src;
  sub_39CF540((__int64 *)&p_src, v22, (__int64)&v22[v23]);
  v11 = *(char **)a2;
  v12 = *(char **)a2;
  if ( p_src == (char *)&src )
  {
    v19 = n;
    if ( n )
    {
      if ( n == 1 )
        *v11 = src;
      else
        memcpy(v11, &src, n);
      v19 = n;
      v11 = *(char **)a2;
    }
    goto LABEL_17;
  }
  v13 = n;
  v14 = src;
  if ( v11 == (char *)(a2 + 16) )
  {
    *(_QWORD *)a2 = p_src;
    *(_QWORD *)(a2 + 8) = v13;
    *(_QWORD *)(a2 + 16) = v14;
    goto LABEL_19;
  }
  v15 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)a2 = p_src;
  *(_QWORD *)(a2 + 8) = v13;
  *(_QWORD *)(a2 + 16) = v14;
  if ( !v12 )
  {
LABEL_19:
    p_src = (char *)&src;
    v12 = (char *)&src;
    goto LABEL_12;
  }
  p_src = v12;
  src = v15;
LABEL_12:
  n = 0;
  *v12 = 0;
  if ( p_src != (char *)&src )
    j_j___libc_free_0((unsigned __int64)p_src);
  v16 = sub_16E4250(v10);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v16 + 16);
    v18 = *(_QWORD *)(v16 + 24);
    *(_QWORD *)(a2 + 32) = v17;
    *(_QWORD *)(a2 + 40) = v18;
  }
}
