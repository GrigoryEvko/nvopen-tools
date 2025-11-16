// Function: sub_EDCB10
// Address: 0xedcb10
//
__int64 *__fastcall sub_EDCB10(__int64 *a1, __int64 a2, unsigned __int64 *a3, __int64 a4, _QWORD *a5, int a6)
{
  unsigned __int64 *v6; // rbx
  unsigned __int64 v8; // r12
  void **v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  void **v12; // rdi
  size_t v13; // r9
  void **v14; // r14
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rbx
  char *v17; // rdi
  const char *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v23; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+28h] [rbp-68h]
  size_t v26; // [rsp+28h] [rbp-68h]
  const char *v27; // [rsp+30h] [rbp-60h] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h]
  __int64 v29; // [rsp+40h] [rbp-50h]
  _BYTE dest[8]; // [rsp+48h] [rbp-48h] BYREF
  char v31; // [rsp+50h] [rbp-40h]
  char v32; // [rsp+51h] [rbp-3Fh]

  if ( a4 )
  {
    v6 = a3;
    v8 = (unsigned __int64)a3 + a4;
    v23 = *(_QWORD *)(a2 + 16);
    while ( 1 )
    {
      if ( v8 <= (unsigned __int64)v6 )
      {
        v27 = 0;
        *a1 = 1;
        sub_9C66B0((__int64 *)&v27);
        return a1;
      }
      if ( v8 - (unsigned __int64)v6 <= 7 )
      {
        v32 = 1;
        v19 = "not enough data to read binary id length";
        goto LABEL_25;
      }
      v13 = *v6;
      v14 = (void **)(v6 + 1);
      v15 = _byteswap_uint64(*v6);
      if ( a6 != 1 )
        v13 = v15;
      if ( !v13 )
      {
        LODWORD(v27) = 9;
        sub_ED89C0(a1, (int *)&v27, "binary id length is 0");
        return a1;
      }
      v16 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 - (unsigned __int64)v14 < v16 )
        break;
      v28 = 0;
      v27 = dest;
      v17 = dest;
      v29 = 10;
      if ( v13 > 0xA )
      {
        v26 = v13;
        sub_C8D290((__int64)&v27, dest, v13, 1u, (__int64)a5, v13);
        v13 = v26;
        v17 = (char *)&v27[v28];
      }
      v9 = v14;
      v25 = v13;
      memcpy(v17, v14, v13);
      v12 = (void **)a5[1];
      v28 += v25;
      if ( v12 == (void **)a5[2] )
      {
        v9 = v12;
        sub_EDC8A0(a5, (__int64)v12, (__int64)&v27, v11);
      }
      else
      {
        if ( v12 )
        {
          v12[1] = 0;
          *v12 = v12 + 3;
          v12[2] = (void *)10;
          if ( v28 )
          {
            v9 = (void **)&v27;
            sub_ED6290((__int64)v12, (char **)&v27, v10, v11, (__int64)a5, v25);
          }
          v12 = (void **)a5[1];
        }
        a5[1] = v12 + 5;
      }
      if ( v27 != dest )
        _libc_free(v27, v9);
      v6 = (unsigned __int64 *)((char *)v14 + v16);
      if ( (unsigned __int64)v6 > v23 )
      {
        v32 = 1;
        v27 = "binary id section is greater than buffer size";
        v31 = 3;
        v20 = sub_22077B0(48);
        v21 = v20;
        if ( v20 )
          goto LABEL_26;
        goto LABEL_27;
      }
    }
    v32 = 1;
    v19 = "not enough data to read binary id data";
LABEL_25:
    v27 = v19;
    v31 = 3;
    v20 = sub_22077B0(48);
    v21 = v20;
    if ( v20 )
    {
LABEL_26:
      *(_DWORD *)(v20 + 8) = 9;
      *(_QWORD *)v20 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v20 + 16), (void **)&v27);
    }
LABEL_27:
    *a1 = v21 | 1;
    return a1;
  }
  *a1 = 1;
  return a1;
}
