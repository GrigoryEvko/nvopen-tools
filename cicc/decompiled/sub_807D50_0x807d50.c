// Function: sub_807D50
// Address: 0x807d50
//
_DWORD *__fastcall sub_807D50(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r14
  __m128i *v3; // r15
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  _QWORD *v6; // rax
  __m128i *v7; // r13
  __m128i *v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  _BYTE *v11; // rax
  void *v12; // rax
  _QWORD *v13; // rax
  _BYTE *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // rbx
  _QWORD *v23; // rdi
  __int64 v24; // [rsp+0h] [rbp-180h]
  _BYTE *v25; // [rsp+8h] [rbp-178h]
  __int64 v26; // [rsp+18h] [rbp-168h]
  __int64 v27; // [rsp+20h] [rbp-160h]
  __int64 *v29; // [rsp+38h] [rbp-148h]
  unsigned int v30; // [rsp+4Ch] [rbp-134h] BYREF
  int v31[8]; // [rsp+50h] [rbp-130h] BYREF
  int v32[8]; // [rsp+70h] [rbp-110h] BYREF
  _BYTE v33[240]; // [rsp+90h] [rbp-F0h] BYREF

  v1 = *(_QWORD **)(*(_QWORD *)(a1 + 152) + 168LL);
  v27 = *(_QWORD *)(a1 + 152);
  v26 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8LL);
  v29 = sub_7F54F0(a1, 0, 0, &v30);
  sub_7E1740(v29[10], (__int64)v31);
  sub_7F6C60((__int64)v29, v30, (__int64)v33);
  v2 = (_QWORD *)*v1;
  if ( *v1 )
  {
    v3 = 0;
    v4 = 0;
    v5 = 0;
    while ( 1 )
    {
      v7 = v3;
      v8 = sub_7E2270(v2[1]);
      v8[8].m128i_i64[0] = (__int64)v2;
      v3 = v8;
      if ( v7 )
      {
        v7[7].m128i_i64[0] = (__int64)v8;
        v6 = sub_73E830((__int64)v8);
        if ( v5 )
          goto LABEL_4;
      }
      else
      {
        v29[5] = (__int64)v8;
        v6 = sub_73E830((__int64)v8);
        if ( v5 )
        {
LABEL_4:
          v4[2] = v6;
          v2 = (_QWORD *)*v2;
          if ( !v2 )
            goto LABEL_9;
          goto LABEL_5;
        }
      }
      v2 = (_QWORD *)*v2;
      v5 = v6;
      if ( !v2 )
        goto LABEL_9;
LABEL_5:
      v4 = v6;
    }
  }
  v5 = 0;
LABEL_9:
  v24 = sub_7E51A0(a1);
  v9 = sub_73E830(v24);
  v25 = sub_724D50(6);
  sub_72D3B0(a1, (__int64)v25, 1);
  v9[2] = sub_730690((__int64)v25);
  v10 = sub_72BA30(5u);
  v11 = sub_73DBF0(0x3Au, (__int64)v10, (__int64)v9);
  v12 = sub_7F0830(v11);
  sub_7F8BA0((__int64)v12, 0, v31, 0, (__int64)v32, 0);
  v13 = sub_7F88E0(v26, 0);
  v14 = sub_73E110((__int64)v13, *(_QWORD *)(v24 + 120));
  sub_7E6AB0(v24, (__int64)v14, v32);
  v15 = sub_73E830(v24);
  v15[2] = v5;
  v16 = (__int64)v15;
  v17 = sub_7F8700(v27);
  v18 = sub_73DBF0(0x69u, v17, v16);
  v19 = sub_7F8700(v27);
  if ( (unsigned int)sub_8D2600(v19) )
  {
    v23 = v18;
    v18 = 0;
    sub_7E69E0(v23, v31);
  }
  v20 = sub_726B30(8);
  *((_QWORD *)v20 + 6) = v18;
  v21 = (__int64)v20;
  sub_7E6810((__int64)v20, (__int64)v31, 1);
  sub_7E17A0(v21);
  *(_BYTE *)(a1 + 193) |= 0x10u;
  return sub_7FB010((__int64)v29, v30, (__int64)v33);
}
