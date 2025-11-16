// Function: sub_2F0E3F0
// Address: 0x2f0e3f0
//
void __fastcall sub_2F0E3F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int8 *v5; // rdi
  unsigned __int8 *v6; // rax
  size_t v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  size_t v13; // rdx
  _QWORD v14[2]; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE *v15; // [rsp+10h] [rbp-90h] BYREF
  __int64 v16; // [rsp+18h] [rbp-88h]
  _BYTE v17[16]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int8 *v18; // [rsp+30h] [rbp-70h] BYREF
  size_t n; // [rsp+38h] [rbp-68h]
  _QWORD src[12]; // [rsp+40h] [rbp-60h] BYREF

  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v3 = *(_QWORD *)a1;
    v15 = 0;
    v16 = 0;
    (*(void (__fastcall **)(__int64, _BYTE **))(v3 + 224))(a1, &v15);
    v4 = sub_CB0A70(a1);
    if ( v15 )
    {
      v18 = (unsigned __int8 *)src;
      sub_2F07580((__int64 *)&v18, v15, (__int64)&v15[v16]);
      v5 = *(unsigned __int8 **)a2;
      v6 = *(unsigned __int8 **)a2;
      if ( v18 != (unsigned __int8 *)src )
      {
        v7 = n;
        v8 = src[0];
        if ( v6 == (unsigned __int8 *)(a2 + 16) )
        {
          *(_QWORD *)a2 = v18;
          *(_QWORD *)(a2 + 8) = v7;
          *(_QWORD *)(a2 + 16) = v8;
        }
        else
        {
          v9 = *(_QWORD *)(a2 + 16);
          *(_QWORD *)a2 = v18;
          *(_QWORD *)(a2 + 8) = v7;
          *(_QWORD *)(a2 + 16) = v8;
          if ( v6 )
          {
            v18 = v6;
            src[0] = v9;
LABEL_7:
            n = 0;
            *v6 = 0;
            if ( v18 != (unsigned __int8 *)src )
              j_j___libc_free_0((unsigned __int64)v18);
            v10 = sub_CB1000(v4);
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
        v18 = (unsigned __int8 *)src;
        v6 = (unsigned __int8 *)src;
        goto LABEL_7;
      }
      v13 = n;
      if ( n )
      {
        if ( n == 1 )
          *v5 = src[0];
        else
          memcpy(v5, src, n);
        v13 = n;
        v5 = *(unsigned __int8 **)a2;
      }
    }
    else
    {
      LOBYTE(src[0]) = 0;
      v5 = *(unsigned __int8 **)a2;
      v13 = 0;
      v18 = (unsigned __int8 *)src;
    }
    *(_QWORD *)(a2 + 8) = v13;
    v5[v13] = 0;
    v6 = v18;
    goto LABEL_7;
  }
  v17[0] = 0;
  src[3] = 0x100000000LL;
  src[4] = &v15;
  v15 = v17;
  v16 = 0;
  n = 0;
  memset(src, 0, 24);
  v18 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v18, 0, 0, 0);
  sub_CB0A70(a1);
  sub_CB6200((__int64)&v18, *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
  v14[0] = v15;
  v14[1] = v16;
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 224LL))(a1, v14);
  v18 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v18);
  if ( v15 != v17 )
    j_j___libc_free_0((unsigned __int64)v15);
}
