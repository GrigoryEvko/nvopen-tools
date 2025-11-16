// Function: sub_370EAB0
// Address: 0x370eab0
//
__int64 *__fastcall sub_370EAB0(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v9; // cc
  __int16 v10; // dx
  char v11; // al
  _QWORD *v12; // r14
  __int16 v13; // ax
  bool v14; // zf
  unsigned __int64 v16; // rax
  int v17; // esi
  unsigned __int16 v18; // dx
  unsigned __int64 v19; // rdx
  const char *v20; // r8
  _QWORD *v21; // rax
  size_t v22; // r12
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  const char *src; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v28; // [rsp+1Ch] [rbp-B4h] BYREF
  unsigned __int16 v29; // [rsp+1Eh] [rbp-B2h] BYREF
  unsigned __int64 v30; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v32[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v33[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v34[4]; // [rsp+50h] [rbp-80h] BYREF
  __m128i v35[2]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v36; // [rsp+90h] [rbp-40h]

  v9 = a3[1] <= 3u;
  v31 = 0;
  if ( v9 || (v10 = *(_WORD *)(*a3 + 2LL), v11 = 0, v10 != 4611) && v10 != 4614 )
  {
    LODWORD(v31) = 65276;
    v11 = 1;
  }
  BYTE4(v31) = v11;
  v12 = (_QWORD *)(a2 + 16);
  sub_3700D50(v35, a2 + 16, v31, a4, a5, a6);
  if ( (v35[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v35[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v13 = 0;
  if ( a3[1] > 3u )
    v13 = *(_WORD *)(*a3 + 2LL);
  v14 = *(_QWORD *)(a2 + 72) == 0;
  *(_WORD *)(a2 + 8) = v13;
  *(_BYTE *)(a2 + 10) = 1;
  if ( !v14 && !*(_QWORD *)(a2 + 56) && !*(_QWORD *)(a2 + 64) )
  {
    v16 = a3[1];
    if ( v16 <= 3 )
    {
      v17 = 0;
      v18 = 0;
    }
    else
    {
      v17 = *(unsigned __int16 *)(*a3 + 2LL);
      v18 = *(_WORD *)(*a3 + 2LL);
    }
    v28 = v18;
    v29 = v16 - 2;
    v20 = sub_370CB70((_QWORD *)(a2 + 16), v17);
    v21 = v33;
    v22 = v19;
    v32[0] = (unsigned __int64)v33;
    if ( &v20[v19] && !v20 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v35[0].m128i_i64[0] = v19;
    if ( v19 > 0xF )
    {
      src = v20;
      v25 = sub_22409D0((__int64)v32, (unsigned __int64 *)v35, 0);
      v20 = src;
      v32[0] = v25;
      v26 = (_QWORD *)v25;
      v33[0] = v35[0].m128i_i64[0];
    }
    else
    {
      if ( v19 == 1 )
      {
        LOBYTE(v33[0]) = *v20;
LABEL_21:
        v32[1] = v19;
        *((_BYTE *)v21 + v19) = 0;
        v35[0].m128i_i64[0] = (__int64)"Record length";
        v36 = 259;
        sub_370BC10((unsigned __int64 *)v34, v12, &v29, v35);
        v23 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = 0;
          v34[0] = v23 | 1;
          sub_9C6670(a1, v34);
          sub_9C66B0(v34);
LABEL_23:
          sub_2240A30(v32);
          return a1;
        }
        v34[0] = 0;
        sub_9C66B0(v34);
        sub_8FD6D0((__int64)v34, "Record kind: ", v32);
        v36 = 260;
        v35[0].m128i_i64[0] = (__int64)v34;
        sub_370E990(&v30, v12, &v28, v35, (unsigned int)&v30);
        sub_2240A30((unsigned __int64 *)v34);
        v24 = v30 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = 0;
          v30 = v24 | 1;
          sub_9C6670(a1, &v30);
          sub_9C66B0((__int64 *)&v30);
          goto LABEL_23;
        }
        v30 = 0;
        sub_9C66B0((__int64 *)&v30);
        sub_2240A30(v32);
        goto LABEL_10;
      }
      if ( !v19 )
        goto LABEL_21;
      v26 = v33;
    }
    memcpy(v26, v20, v22);
    v19 = v35[0].m128i_i64[0];
    v21 = (_QWORD *)v32[0];
    goto LABEL_21;
  }
LABEL_10:
  *a1 = 1;
  return a1;
}
