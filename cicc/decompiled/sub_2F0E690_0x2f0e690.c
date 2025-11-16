// Function: sub_2F0E690
// Address: 0x2f0e690
//
void __fastcall sub_2F0E690(__int64 a1, __int64 a2)
{
  char *v3; // rdx
  __int64 v4; // rax
  void (__fastcall *v5)(__int64, char **, _QWORD); // rbx
  unsigned int v6; // eax
  __int64 v7; // rax
  void (__fastcall *v8)(__int64, _BYTE **, _QWORD); // r13
  unsigned int v9; // eax
  __int64 v10; // r13
  unsigned __int8 *v11; // rdi
  __int64 *v12; // rax
  size_t v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  size_t v19; // rdx
  char *v20[2]; // [rsp+0h] [rbp-110h] BYREF
  _BYTE *v21; // [rsp+10h] [rbp-100h] BYREF
  __int64 v22; // [rsp+18h] [rbp-F8h]
  __int64 v23; // [rsp+20h] [rbp-F0h]
  __int64 v24; // [rsp+28h] [rbp-E8h]
  __int64 v25; // [rsp+30h] [rbp-E0h]
  __int64 v26; // [rsp+38h] [rbp-D8h]
  __int64 **v27; // [rsp+40h] [rbp-D0h]
  __int64 *p_src; // [rsp+50h] [rbp-C0h] BYREF
  size_t n; // [rsp+58h] [rbp-B8h]
  __int64 src; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v31[168]; // [rsp+68h] [rbp-A8h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v26 = 0x100000000LL;
    v27 = &p_src;
    v21 = &unk_49DD288;
    p_src = (__int64 *)v31;
    n = 0;
    src = 128;
    v22 = 2;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    sub_CB5980((__int64)&v21, 0, 0, 0);
    sub_CB0A70(a1);
    sub_CB6200((__int64)&v21, *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
    v3 = (char *)v27[1];
    v20[0] = (char *)*v27;
    v4 = *(_QWORD *)a1;
    v20[1] = v3;
    v5 = *(void (__fastcall **)(__int64, char **, _QWORD))(v4 + 216);
    v6 = sub_C2FE50(v20[0], (__int64)v3, 1);
    v5(a1, v20, v6);
    v21 = &unk_49DD388;
    sub_CB5840((__int64)&v21);
    if ( p_src != (__int64 *)v31 )
      _libc_free((unsigned __int64)p_src);
    return;
  }
  v7 = *(_QWORD *)a1;
  v21 = 0;
  v22 = 0;
  v8 = *(void (__fastcall **)(__int64, _BYTE **, _QWORD))(v7 + 216);
  v9 = sub_C2FE50(0, 0, 1);
  v8(a1, &v21, v9);
  v10 = sub_CB0A70(a1);
  if ( !v21 )
  {
    LOBYTE(src) = 0;
    v11 = *(unsigned __int8 **)a2;
    v19 = 0;
    p_src = &src;
LABEL_15:
    *(_QWORD *)(a2 + 8) = v19;
    v11[v19] = 0;
    v12 = p_src;
    goto LABEL_10;
  }
  p_src = &src;
  sub_2F07580((__int64 *)&p_src, v21, (__int64)&v21[v22]);
  v11 = *(unsigned __int8 **)a2;
  v12 = *(__int64 **)a2;
  if ( p_src == &src )
  {
    v19 = n;
    if ( n )
    {
      if ( n == 1 )
        *v11 = src;
      else
        memcpy(v11, &src, n);
      v19 = n;
      v11 = *(unsigned __int8 **)a2;
    }
    goto LABEL_15;
  }
  v13 = n;
  v14 = src;
  if ( v11 == (unsigned __int8 *)(a2 + 16) )
  {
    *(_QWORD *)a2 = p_src;
    *(_QWORD *)(a2 + 8) = v13;
    *(_QWORD *)(a2 + 16) = v14;
    goto LABEL_17;
  }
  v15 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)a2 = p_src;
  *(_QWORD *)(a2 + 8) = v13;
  *(_QWORD *)(a2 + 16) = v14;
  if ( !v12 )
  {
LABEL_17:
    p_src = &src;
    v12 = &src;
    goto LABEL_10;
  }
  p_src = v12;
  src = v15;
LABEL_10:
  n = 0;
  *(_BYTE *)v12 = 0;
  if ( p_src != &src )
    j_j___libc_free_0((unsigned __int64)p_src);
  v16 = sub_CB1000(v10);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v16 + 16);
    v18 = *(_QWORD *)(v16 + 24);
    *(_QWORD *)(a2 + 32) = v17;
    *(_QWORD *)(a2 + 40) = v18;
  }
}
