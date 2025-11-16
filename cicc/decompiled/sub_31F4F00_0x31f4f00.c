// Function: sub_31F4F00
// Address: 0x31f4f00
//
void __fastcall sub_31F4F00(__int64 *a1, const void *a2, unsigned __int64 a3, int a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  size_t v8; // rbx
  _BYTE *v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  _BYTE *v13; // [rsp+0h] [rbp-70h] BYREF
  size_t v14; // [rsp+8h] [rbp-68h]
  unsigned __int64 v15; // [rsp+10h] [rbp-60h]
  _BYTE v16[88]; // [rsp+18h] [rbp-58h] BYREF

  v6 = (unsigned int)(65279 - a4);
  if ( v6 < a3 )
  {
    v14 = 0;
    v13 = v16;
    v15 = 32;
LABEL_3:
    sub_C8D290((__int64)&v13, v16, v6, 1u, a5, a6);
    v8 = v6;
    v9 = &v13[v14];
    goto LABEL_4;
  }
  v14 = 0;
  v8 = a3;
  v13 = v16;
  v15 = 32;
  if ( a3 > 0x20 )
  {
    v6 = a3;
    goto LABEL_3;
  }
  if ( !a3 )
    goto LABEL_6;
  v9 = v16;
LABEL_4:
  memcpy(v9, a2, v8);
  v8 += v14;
  v14 = v8;
  if ( v8 + 1 > v15 )
  {
    sub_C8D290((__int64)&v13, v16, v8 + 1, 1u, v10, v11);
    v8 = v14;
  }
LABEL_6:
  v13[v8] = 0;
  v12 = *a1;
  ++v14;
  (*(void (__fastcall **)(__int64 *, _BYTE *))(v12 + 512))(a1, v13);
  if ( v13 != v16 )
    _libc_free((unsigned __int64)v13);
}
