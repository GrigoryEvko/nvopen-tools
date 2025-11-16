// Function: sub_39D1470
// Address: 0x39d1470
//
void __fastcall sub_39D1470(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  char *v5; // rdi
  char *v6; // rax
  size_t v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  size_t v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v17; // [rsp+10h] [rbp-70h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  _BYTE v19[16]; // [rsp+20h] [rbp-60h] BYREF
  char *p_src; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  __int64 src; // [rsp+40h] [rbp-40h] BYREF
  __int64 v23; // [rsp+48h] [rbp-38h]
  int v24; // [rsp+50h] [rbp-30h]
  __int64 *v25; // [rsp+58h] [rbp-28h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v3 = *(_QWORD *)a1;
    v17 = 0;
    v18 = 0;
    (*(void (__fastcall **)(__int64, _BYTE **))(v3 + 224))(a1, &v17);
    v4 = sub_16E4080(a1);
    if ( v17 )
    {
      p_src = (char *)&src;
      sub_39CF540((__int64 *)&p_src, v17, (__int64)&v17[v18]);
      v5 = *(char **)a2;
      v6 = *(char **)a2;
      if ( p_src != (char *)&src )
      {
        v7 = n;
        v8 = src;
        if ( v6 == (char *)(a2 + 16) )
        {
          *(_QWORD *)a2 = p_src;
          *(_QWORD *)(a2 + 8) = v7;
          *(_QWORD *)(a2 + 16) = v8;
        }
        else
        {
          v9 = *(_QWORD *)(a2 + 16);
          *(_QWORD *)a2 = p_src;
          *(_QWORD *)(a2 + 8) = v7;
          *(_QWORD *)(a2 + 16) = v8;
          if ( v6 )
          {
            p_src = v6;
            src = v9;
LABEL_7:
            n = 0;
            *v6 = 0;
            if ( p_src != (char *)&src )
              j_j___libc_free_0((unsigned __int64)p_src);
            v10 = sub_16E4250(v4);
            if ( v10 )
            {
              v11 = *(_QWORD *)(v10 + 16);
              v12 = *(_QWORD *)(v10 + 24);
              *(_QWORD *)(a2 + 32) = v11;
              *(_QWORD *)(a2 + 40) = v12;
            }
            return;
          }
        }
        p_src = (char *)&src;
        v6 = (char *)&src;
        goto LABEL_7;
      }
      v13 = n;
      if ( n )
      {
        if ( n == 1 )
          *v5 = src;
        else
          memcpy(v5, &src, n);
        v13 = n;
        v5 = *(char **)a2;
      }
    }
    else
    {
      LOBYTE(src) = 0;
      v5 = *(char **)a2;
      v13 = 0;
      p_src = (char *)&src;
    }
    *(_QWORD *)(a2 + 8) = v13;
    v5[v13] = 0;
    v6 = p_src;
    goto LABEL_7;
  }
  v19[0] = 0;
  v17 = v19;
  v18 = 0;
  v24 = 1;
  p_src = (char *)&unk_49EFBE0;
  v23 = 0;
  src = 0;
  n = 0;
  v25 = (__int64 *)&v17;
  sub_16E4080(a1);
  sub_16E7EE0((__int64)&p_src, *(char **)a2, *(_QWORD *)(a2 + 8));
  if ( v23 != n )
    sub_16E7BA0((__int64 *)&p_src);
  v14 = *v25;
  v16[1] = v25[1];
  v15 = *(_QWORD *)a1;
  v16[0] = v14;
  (*(void (__fastcall **)(__int64, _QWORD *))(v15 + 224))(a1, v16);
  sub_16E7BC0((__int64 *)&p_src);
  if ( v17 != v19 )
    j_j___libc_free_0((unsigned __int64)v17);
}
