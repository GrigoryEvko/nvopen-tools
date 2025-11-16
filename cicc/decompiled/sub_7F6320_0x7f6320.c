// Function: sub_7F6320
// Address: 0x7f6320
//
_QWORD *__fastcall sub_7F6320(__int64 a1, const __m128i *a2, __int64 a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  const __m128i *v5; // rbx
  _QWORD *i; // r13
  _QWORD *v7; // rax
  const __m128i *v10; // rdi
  void *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  int v16; // [rsp+4h] [rbp-2Ch] BYREF
  int v17; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v18[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(_QWORD **)(a1 + 40);
  if ( !v4 )
    return v4;
  v4 = (_QWORD *)v4[2];
  if ( !v4 )
    return v4;
  if ( (*(_BYTE *)a1 & 2) != 0 )
  {
    v5 = (const __m128i *)a2[1].m128i_i64[0];
    if ( !v5 )
      return 0;
    v4 = sub_7E8090(v5, 1u);
    for ( i = v4; ; i = v7 )
    {
      v5 = (const __m128i *)v5[1].m128i_i64[0];
      if ( !v5 )
        break;
      v7 = sub_7E8090(v5, 1u);
      if ( v4 )
        i[2] = v7;
      else
        v4 = v7;
    }
    return v4;
  }
  sub_87ADD0(v4, &v16, &v17, v18);
  if ( v18[0] )
  {
    v12 = sub_8865A0("destroying_delete_t");
    v13 = sub_7E7CB0(*(_QWORD *)(v12 + 88));
    v4 = sub_73E830(v13);
    v3 = v4;
    if ( v16 )
    {
      v14 = sub_7E8090(a2, 1u);
      v3 = v14;
      if ( v4 )
        v4[2] = v14;
      else
        v4 = v14;
    }
  }
  else if ( v16 )
  {
    v4 = sub_7E8090(a2, 1u);
    v3 = v4;
  }
  else
  {
    v4 = 0;
  }
  if ( !v17 )
    return v4;
  v10 = (const __m128i *)a2[1].m128i_i64[0];
  if ( v10 )
  {
    v11 = sub_7E8090(v10, 1u);
  }
  else
  {
    v15 = sub_7F6130(a3);
    v11 = sub_73E130(v15, unk_4F06C60);
  }
  if ( !v4 )
    return v11;
  v3[2] = v11;
  return v4;
}
