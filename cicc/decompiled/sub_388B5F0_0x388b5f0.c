// Function: sub_388B5F0
// Address: 0x388b5f0
//
__int64 __fastcall sub_388B5F0(__int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v3; // r13
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  _BYTE *v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rcx
  size_t v9; // rsi
  __int64 v10; // rdi
  size_t v11; // rdx
  _QWORD *v12; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after source_filename") )
    return 1;
  v1 = sub_388B0A0(a1, (unsigned __int64 *)(a1 + 1464));
  if ( (_BYTE)v1 )
    return 1;
  v3 = *(_QWORD **)(a1 + 176);
  if ( !v3 )
    return v1;
  v4 = *(_BYTE **)(a1 + 1464);
  if ( !v4 )
  {
    LOBYTE(src[0]) = 0;
    v6 = (_BYTE *)v3[26];
    v11 = 0;
    v12 = src;
LABEL_14:
    v3[27] = v11;
    v6[v11] = 0;
    v7 = v12;
    goto LABEL_11;
  }
  v5 = *(_QWORD *)(a1 + 1472);
  v12 = src;
  sub_3887410((__int64 *)&v12, v4, (__int64)&v4[v5]);
  v6 = (_BYTE *)v3[26];
  v7 = v6;
  if ( v12 == src )
  {
    v11 = n;
    if ( n )
    {
      if ( n == 1 )
        *v6 = src[0];
      else
        memcpy(v6, src, n);
      v11 = n;
      v6 = (_BYTE *)v3[26];
    }
    goto LABEL_14;
  }
  v8 = src[0];
  v9 = n;
  if ( v6 == (_BYTE *)(v3 + 28) )
  {
    v3[26] = v12;
    v3[27] = v9;
    v3[28] = v8;
  }
  else
  {
    v10 = v3[28];
    v3[26] = v12;
    v3[27] = v9;
    v3[28] = v8;
    if ( v7 )
    {
      v12 = v7;
      src[0] = v10;
      goto LABEL_11;
    }
  }
  v12 = src;
  v7 = src;
LABEL_11:
  n = 0;
  *(_BYTE *)v7 = 0;
  if ( v12 != src )
    j_j___libc_free_0((unsigned __int64)v12);
  return v1;
}
