// Function: sub_BA89D0
// Address: 0xba89d0
//
__int64 *__fastcall sub_BA89D0(__int64 *a1, __int64 a2, const void *a3, size_t a4)
{
  const void *v4; // r8
  size_t v7; // rax
  __int64 v8; // rdi
  _BYTE *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r12
  size_t v12; // rdx
  size_t v13; // rbx
  unsigned __int64 v14; // rdx
  _BYTE *v15; // r15
  size_t v16; // r12
  __int64 v17; // rax
  __int64 v18; // rbx
  _BYTE *v19; // rdi
  _BYTE *v21; // rdi
  _BYTE *v23; // [rsp+8h] [rbp-78h]
  _BYTE *v24; // [rsp+10h] [rbp-70h] BYREF
  size_t v25; // [rsp+18h] [rbp-68h]
  unsigned __int64 v26; // [rsp+20h] [rbp-60h]
  _BYTE dest[88]; // [rsp+28h] [rbp-58h] BYREF

  v4 = a3;
  v24 = dest;
  v25 = 0;
  v26 = 32;
  if ( a4 > 0x20 )
  {
    sub_C8D290(&v24, dest, a4, 1);
    v4 = a3;
    v21 = &v24[v25];
  }
  else
  {
    v7 = a4;
    if ( !a4 )
      goto LABEL_3;
    v21 = dest;
  }
  memcpy(v21, v4, a4);
  v7 = a4 + v25;
LABEL_3:
  v8 = *(_QWORD *)(a2 + 168);
  v9 = *(_BYTE **)(a2 + 176);
  v25 = v7;
  v10 = (_BYTE *)sub_C80C60(v8, v9, 0, a4, v4);
  v11 = v25;
  v13 = v12;
  v14 = v25 + v12;
  if ( v14 > v26 )
  {
    v9 = dest;
    v23 = v10;
    sub_C8D290(&v24, dest, v14, 1);
    v11 = v25;
    v10 = v23;
  }
  v15 = v24;
  if ( v13 )
  {
    v9 = v10;
    memcpy(&v24[v11], v10, v13);
    v15 = v24;
    v11 = v25;
  }
  v16 = v13 + v11;
  v25 = v16;
  v17 = sub_22077B0(2504);
  v18 = v17;
  if ( v17 )
  {
    v9 = v15;
    sub_C88C40(v17, v15, v16);
  }
  v19 = v24;
  *a1 = v18;
  if ( v19 != dest )
    _libc_free(v19, v9);
  return a1;
}
