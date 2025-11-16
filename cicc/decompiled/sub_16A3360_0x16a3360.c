// Function: sub_16A3360
// Address: 0x16a3360
//
__int64 __fastcall sub_16A3360(__int64 a1, __int16 *a2, unsigned int a3, bool *a4)
{
  __int16 *v5; // r13
  __int16 *v8; // rax
  __int64 v10; // rsi
  _QWORD *v11; // r8
  __int64 v12; // r12
  __int64 i; // rbx
  _QWORD *v14; // [rsp+8h] [rbp-A8h]
  _QWORD *v15; // [rsp+10h] [rbp-A0h]
  _QWORD *v16; // [rsp+10h] [rbp-A0h]
  _QWORD *v17; // [rsp+10h] [rbp-A0h]
  unsigned int v18; // [rsp+1Ch] [rbp-94h]
  _BYTE v19[32]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v20; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-68h]
  void *v22[9]; // [rsp+68h] [rbp-48h] BYREF

  v5 = *(__int16 **)(a1 + 8);
  if ( a2 == v5 )
  {
    *a4 = 0;
    return 0;
  }
  v8 = (__int16 *)sub_16982C0();
  if ( v8 == v5 )
  {
    v18 = sub_16995F0(*(_QWORD *)(a1 + 16) + 8LL, a2, a3, a4);
    v10 = a1 + 8;
    if ( v5 == *(__int16 **)(a1 + 8) )
      v10 = *(_QWORD *)(a1 + 16) + 8LL;
    v14 = (_QWORD *)(a1 + 8);
    sub_1698450((__int64)v19, v10);
    sub_1698450((__int64)&v20, (__int64)v19);
    sub_169E320(v22, &v20, a2);
    sub_1698460((__int64)&v20);
    v15 = (_QWORD *)(a1 + 8);
    v11 = (_QWORD *)(a1 + 8);
    if ( v5 == *(__int16 **)(a1 + 8) )
    {
      if ( v5 == v22[0] )
      {
        v12 = *(_QWORD *)(a1 + 16);
        if ( v12 )
        {
          for ( i = v12 + 32LL * *(_QWORD *)(v12 - 8); v12 != i; v11 = v16 )
          {
            i -= 32;
            v16 = v11;
            if ( v5 == *(__int16 **)(i + 8) )
              sub_169DEB0((__int64 *)(i + 16));
            else
              sub_1698460(i + 8);
          }
          v17 = v11;
          j_j_j___libc_free_0_0(v12 - 8);
          v11 = v17;
        }
        sub_169C7E0(v11, v22);
        goto LABEL_13;
      }
    }
    else if ( v5 != v22[0] )
    {
      sub_16983E0((__int64)v15, (__int64)v22);
LABEL_13:
      sub_127D120(v22);
      sub_1698460((__int64)v19);
      return v18;
    }
    sub_127D120(v15);
    if ( v5 == v22[0] )
      sub_169C7E0(v14, v22);
    else
      sub_1698450((__int64)v14, (__int64)v22);
    goto LABEL_13;
  }
  if ( a2 != v8 )
    return sub_16995F0(a1 + 8, a2, a3, a4);
  v18 = sub_16995F0(a1 + 8, word_42AE980, a3, a4);
  sub_169D7E0((__int64)&v20, (__int64 *)(a1 + 8));
  sub_169D060(v22, (__int64)a2, &v20);
  sub_169ED90((void **)(a1 + 8), v22);
  sub_127D120(v22);
  if ( v21 > 0x40 )
  {
    if ( v20 )
      j_j___libc_free_0_0(v20);
  }
  return v18;
}
