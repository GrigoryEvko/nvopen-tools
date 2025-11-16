// Function: sub_CCA350
// Address: 0xcca350
//
__int64 __fastcall sub_CCA350(__int64 *a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  size_t v12; // rdi
  size_t v13; // rdx
  size_t v14; // rbx
  const void *v15; // r15
  unsigned __int64 v16; // rdx
  size_t v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  size_t v21; // rdi
  size_t v22; // rdx
  size_t v23; // rbx
  const void *v24; // r15
  unsigned __int64 v25; // rdx
  void *v26; // rax
  __int64 result; // rax
  _BYTE *v28; // rdi
  __int64 v29; // r8
  __int64 v30; // r9
  void *v31[4]; // [rsp+0h] [rbp-C0h] BYREF
  __int16 v32; // [rsp+20h] [rbp-A0h]
  _BYTE *v33; // [rsp+30h] [rbp-90h] BYREF
  size_t v34; // [rsp+38h] [rbp-88h]
  unsigned __int64 v35; // [rsp+40h] [rbp-80h]
  _BYTE dest[120]; // [rsp+48h] [rbp-78h] BYREF

  v33 = dest;
  v34 = 0;
  v35 = 64;
  if ( a3 > 0x40 )
  {
    sub_C8D290((__int64)&v33, dest, a3, 1u, a5, a6);
    v28 = &v33[v34];
  }
  else
  {
    v8 = a3;
    if ( !a3 )
      goto LABEL_3;
    v28 = dest;
  }
  memcpy(v28, a2, a3);
  v8 = a3 + v34;
  v34 = v8;
  if ( v35 < v8 + 1 )
  {
    sub_C8D290((__int64)&v33, dest, v8 + 1, 1u, v29, v30);
    v8 = v34;
  }
LABEL_3:
  v33[v8] = 45;
  ++v34;
  v9 = sub_CC72D0(a1);
  v12 = v34;
  v14 = v13;
  v15 = (const void *)v9;
  v16 = v34 + v13;
  if ( v16 > v35 )
  {
    sub_C8D290((__int64)&v33, dest, v16, 1u, v10, v11);
    v12 = v34;
  }
  if ( v14 )
  {
    memcpy(&v33[v12], v15, v14);
    v12 = v34;
  }
  v17 = v12 + v14;
  v34 = v17;
  if ( v17 + 1 > v35 )
  {
    sub_C8D290((__int64)&v33, dest, v17 + 1, 1u, v10, v11);
    v17 = v34;
  }
  v33[v17] = 45;
  ++v34;
  v18 = sub_CC75D0(a1);
  v21 = v34;
  v23 = v22;
  v24 = (const void *)v18;
  v25 = v34 + v22;
  if ( v25 > v35 )
  {
    sub_C8D290((__int64)&v33, dest, v25, 1u, v19, v20);
    v21 = v34;
  }
  v26 = v33;
  if ( v23 )
  {
    memcpy(&v33[v21], v24, v23);
    v26 = v33;
    v21 = v34;
  }
  v31[0] = v26;
  v34 = v21 + v23;
  v32 = 261;
  v31[1] = (void *)(v21 + v23);
  result = sub_CCA250(a1, v31);
  if ( v33 != dest )
    return _libc_free(v33, v31);
  return result;
}
