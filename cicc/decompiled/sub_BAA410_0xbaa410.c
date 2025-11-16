// Function: sub_BAA410
// Address: 0xbaa410
//
__int64 __fastcall sub_BAA410(__int64 a1, void *a2, size_t a3)
{
  unsigned int v4; // eax
  size_t v5; // r14
  unsigned __int64 v6; // rsi
  __int64 v7; // rbx
  size_t *v9; // rbx
  __int64 *v10; // rax
  void *src; // [rsp+8h] [rbp-D8h]
  size_t **v12; // [rsp+10h] [rbp-D0h]
  unsigned int v13; // [rsp+1Ch] [rbp-C4h]
  _BYTE v14[24]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+38h] [rbp-A8h]
  char v16; // [rsp+4Ch] [rbp-94h]
  void *v17; // [rsp+60h] [rbp-80h]
  size_t n; // [rsp+68h] [rbp-78h]
  _BYTE v19[24]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v20; // [rsp+88h] [rbp-58h]
  char v21; // [rsp+9Ch] [rbp-44h]

  sub_AA87D0((__int64)v14);
  v17 = a2;
  n = a3;
  sub_AA87A0((__int64)v19, (__int64)v14);
  v4 = sub_C92610(v17, n);
  v5 = n;
  v6 = (unsigned __int64)v17;
  src = v17;
  v13 = sub_C92740(a1 + 128, v17, n, v4);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 8LL * v13);
  v12 = (size_t **)(*(_QWORD *)(a1 + 128) + 8LL * v13);
  if ( v7 )
  {
    if ( v7 != -8 )
    {
      if ( v21 )
        goto LABEL_4;
LABEL_13:
      _libc_free(v20, v6);
      if ( v16 )
        goto LABEL_5;
LABEL_14:
      _libc_free(v15, v6);
      goto LABEL_5;
    }
    --*(_DWORD *)(a1 + 144);
  }
  v9 = (size_t *)sub_C7D670(v5 + 73, 8);
  if ( v5 )
    memcpy(v9 + 9, src, v5);
  *((_BYTE *)v9 + v5 + 72) = 0;
  *v9 = v5;
  sub_AA87A0((__int64)(v9 + 1), (__int64)v19);
  v6 = v13;
  *v12 = v9;
  ++*(_DWORD *)(a1 + 140);
  v10 = (__int64 *)(*(_QWORD *)(a1 + 128) + 8LL * (unsigned int)sub_C929D0(a1 + 128, v13));
  v7 = *v10;
  if ( *v10 )
    goto LABEL_11;
  do
  {
    do
    {
      v7 = v10[1];
      ++v10;
    }
    while ( !v7 );
LABEL_11:
    ;
  }
  while ( v7 == -8 );
  if ( !v21 )
    goto LABEL_13;
LABEL_4:
  if ( !v16 )
    goto LABEL_14;
LABEL_5:
  *(_QWORD *)(v7 + 8) = v7;
  return v7 + 8;
}
