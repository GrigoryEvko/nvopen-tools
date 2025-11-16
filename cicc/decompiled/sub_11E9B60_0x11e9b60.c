// Function: sub_11E9B60
// Address: 0x11e9b60
//
__int64 __fastcall sub_11E9B60(__int64 a1, __int64 *a2, __int64 a3, size_t a4, __int64 a5, __int64 a6)
{
  const void *v6; // r8
  size_t v10; // rax
  __int64 *v11; // rsi
  unsigned int v12; // eax
  unsigned int v13; // r12d
  _BYTE *v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  _BYTE *v19; // [rsp+10h] [rbp-60h] BYREF
  size_t v20; // [rsp+18h] [rbp-58h]
  unsigned __int64 v21; // [rsp+20h] [rbp-50h]
  _BYTE dest[72]; // [rsp+28h] [rbp-48h] BYREF

  v6 = (const void *)a3;
  v19 = dest;
  v20 = 0;
  v21 = 20;
  if ( a4 > 0x14 )
  {
    sub_C8D290((__int64)&v19, dest, a4, 1u, a3, a6);
    v6 = (const void *)a3;
    v15 = &v19[v20];
  }
  else
  {
    v10 = a4;
    if ( !a4 )
      goto LABEL_3;
    v15 = dest;
  }
  memcpy(v15, v6, a4);
  v10 = a4 + v20;
  v20 = v10;
  if ( v21 < v10 + 1 )
  {
    sub_C8D290((__int64)&v19, dest, v10 + 1, 1u, v16, v17);
    v10 = v20;
  }
LABEL_3:
  v19[v10] = 102;
  v11 = *(__int64 **)(a1 + 24);
  LOBYTE(v12) = sub_11C9D10(a2, v11, v19, ++v20);
  v13 = v12;
  if ( v19 != dest )
    _libc_free(v19, v11);
  return v13;
}
