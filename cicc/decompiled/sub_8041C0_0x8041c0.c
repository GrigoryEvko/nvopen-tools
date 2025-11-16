// Function: sub_8041C0
// Address: 0x8041c0
//
__int64 __fastcall sub_8041C0(__m128i *a1)
{
  __int8 v2; // bl
  __int64 v3; // r14
  __int64 v4; // rdi
  char v5; // r15
  bool v6; // bl
  __int64 v7; // rax
  __int64 v8; // r13
  _BOOL4 *v9; // rcx
  __m128i *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 result; // rax
  __m128i *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rax
  _BYTE *v21; // rax
  const __m128i *v22; // rax
  const __m128i *v23; // rax
  __m128i *v24; // rax
  char v25; // al
  __m128i *v26; // rdi
  __int64 v27; // rax
  _BOOL4 *v28; // [rsp+8h] [rbp-C8h]
  _BOOL4 *v29; // [rsp+10h] [rbp-C0h]
  char v30; // [rsp+1Ah] [rbp-B6h]
  bool v31; // [rsp+1Bh] [rbp-B5h]
  int v32; // [rsp+1Ch] [rbp-B4h]
  bool v33; // [rsp+1Ch] [rbp-B4h]
  bool v34; // [rsp+1Ch] [rbp-B4h]
  int v35; // [rsp+28h] [rbp-A8h] BYREF
  unsigned int v36; // [rsp+2Ch] [rbp-A4h] BYREF
  __m128i v37[2]; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v38[4]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v39; // [rsp+90h] [rbp-40h]

  v2 = a1[1].m128i_i8[9];
  v3 = a1[3].m128i_i64[1];
  v36 = 0;
  v32 = v2 & 1;
  v31 = !(v2 & 1);
  if ( *(_BYTE *)(v3 + 48) != 3 || (v2 & 1) != 0 || *(_QWORD *)(v3 + 16) || (*(_BYTE *)(v3 + 51) & 0xA) != 0 )
  {
    v4 = a1->m128i_i64[0];
    v5 = 0;
    v6 = (v2 & 4) != 0;
    if ( !dword_4D047EC )
      goto LABEL_4;
  }
  else
  {
    v7 = *(_QWORD *)(v3 + 112);
    if ( v7 )
    {
      v4 = a1->m128i_i64[0];
      v5 = 0;
      v6 = (v2 & 4) != 0;
      if ( !dword_4D047EC )
        goto LABEL_5;
    }
    else
    {
      v26 = *(__m128i **)(v3 + 56);
      if ( v26[1].m128i_i8[8] != 2 )
      {
        sub_7EE560(v26, 0);
        return sub_730620((__int64)a1, *(const __m128i **)(v3 + 56));
      }
      v4 = a1->m128i_i64[0];
      v5 = 0;
      v6 = (v2 & 4) != 0;
      if ( !dword_4D047EC )
        goto LABEL_45;
    }
  }
  v5 = 0;
  if ( !(unsigned int)sub_8DD010(v4) || *(_BYTE *)(v4 + 140) == 12 && (v5 = 1, *(_QWORD *)(v4 + 8)) )
  {
LABEL_4:
    v7 = *(_QWORD *)(v3 + 112);
    if ( !v7 )
      goto LABEL_45;
LABEL_5:
    v8 = *(_QWORD *)(v7 + 8);
    v9 = 0;
    goto LABEL_6;
  }
  v5 = 1;
  sub_8DD360(v4);
  v7 = *(_QWORD *)(v3 + 112);
  if ( v7 )
    goto LABEL_5;
LABEL_45:
  v20 = sub_7E9260(v4, v3, &v35);
  v9 = 0;
  v8 = v20;
  if ( !v35 )
    v9 = (_BOOL4 *)&v36;
LABEL_6:
  if ( v8 )
  {
    *(_QWORD *)(v3 + 8) = v8;
    if ( v5 )
      *(_BYTE *)(v8 + 173) |= 2u;
    v28 = v9;
    sub_7264E0((__int64)a1, 3);
    a1[3].m128i_i64[1] = v8;
    sub_7F9080(v8, (__int64)v38);
    v30 = *(_BYTE *)(v3 + 48);
    v33 = v30 == 5;
    sub_7E1780((__int64)a1, (__int64)v37);
    sub_802F60(v3, a1->m128i_i64[0], v37);
    v10 = v38;
    sub_7FEC50(v3, v38, 0, 0, 0, 0, v37, v28, 0);
    if ( (*(_BYTE *)(v8 + 173) & 2) != 0 )
    {
      v21 = sub_726B30(22);
      v21[72] = 0;
      *((_QWORD *)v21 + 10) = v8;
      sub_7FCA00(v21);
    }
    v14 = v36;
    if ( v36 )
    {
      v10 = (__m128i *)v3;
      sub_7FCA60(*(_QWORD *)(v3 + 8), v3);
      if ( *(_BYTE *)(v8 + 177) != 3 )
      {
LABEL_13:
        result = v33;
        LOBYTE(result) = v6 || v33;
        if ( !v6 && !v33 )
          goto LABEL_14;
        goto LABEL_17;
      }
    }
    else if ( *(_BYTE *)(v8 + 177) != 3 )
    {
      goto LABEL_13;
    }
    if ( *(_BYTE *)(v8 + 136) <= 2u )
    {
      v10 = (__m128i *)(v8 + 177);
      sub_7EC360(v8, (__m128i *)(v8 + 177), (__int64 *)(v8 + 184));
      result = v33;
      LOBYTE(result) = v6 || v33;
      if ( !v6 && !v33 )
        goto LABEL_14;
    }
    else
    {
      v10 = a1;
      sub_7FBBC0(v8, (__int64)a1);
      result = v33;
      LOBYTE(result) = v6 || v33;
      if ( !v6 && !v33 )
        goto LABEL_14;
    }
  }
  else
  {
    v29 = v9;
    v23 = *(const __m128i **)(*(_QWORD *)(v3 + 112) + 88LL);
    v38[0] = _mm_loadu_si128(v23);
    v38[1] = _mm_loadu_si128(v23 + 1);
    v38[2] = _mm_loadu_si128(v23 + 2);
    v38[3] = _mm_loadu_si128(v23 + 3);
    v39 = v23[4].m128i_i64[0];
    v24 = sub_7F9430((__int64)v38, v32, 0);
    sub_730620((__int64)a1, v24);
    v25 = *(_BYTE *)(v3 + 48);
    *(_QWORD *)(v3 + 112) = 0;
    v30 = v25;
    v34 = v25 == 5;
    sub_7E1780((__int64)a1, (__int64)v37);
    sub_802F60(v3, a1->m128i_i64[0], v37);
    v10 = v38;
    sub_7FEC50(v3, v38, 0, 0, 0, 0, v37, v29, 0);
    result = v34;
    LOBYTE(result) = v6 || v34;
    if ( !v6 && !v34 )
      goto LABEL_14;
  }
LABEL_17:
  if ( a1[1].m128i_i8[8] == 1 && a1[3].m128i_i8[8] == 91 )
  {
    v16 = (__m128i *)a1[4].m128i_i64[1];
    result = v16[1].m128i_i64[0];
    if ( *(_BYTE *)(result + 24) != 1 || *(_BYTE *)(result + 56) != 91 )
    {
      if ( !v6 )
      {
        if ( v30 != 5 )
          goto LABEL_14;
        result = sub_8D32B0(v16->m128i_i64[0]);
        if ( !(_DWORD)result )
          goto LABEL_14;
        if ( v16[1].m128i_i8[8] != 1 )
          goto LABEL_14;
        if ( v16[3].m128i_i8[8] != 105 )
          goto LABEL_14;
        result = v16[4].m128i_i64[1];
        if ( *(_BYTE *)(result + 24) != 20 )
          goto LABEL_14;
        result = *(_QWORD *)(result + 56);
        if ( *(_BYTE *)(result + 174) != 1 )
          goto LABEL_14;
        v17 = sub_8D46C0(v16->m128i_i64[0]);
        v18 = a1->m128i_i64[0];
        v19 = v17;
        v10 = (__m128i *)a1->m128i_i64[0];
        result = sub_8DEFB0(v17, a1->m128i_i64[0], 1, 0);
        if ( !(_DWORD)result )
          goto LABEL_14;
        v16[1].m128i_i8[9] &= ~4u;
        if ( v19 != v18 && !(unsigned int)sub_8D97D0(v18, v19, 1, v14, v12) )
        {
          v16[1].m128i_i64[0] = 0;
          v27 = sub_72D2E0(a1->m128i_i64[0]);
          v16 = (__m128i *)sub_73E110((__int64)v16, v27);
        }
        v16 = (__m128i *)sub_73DCD0(v16);
      }
      v10 = v16;
      result = sub_730620((__int64)a1, v16);
    }
  }
LABEL_14:
  if ( (a1[1].m128i_i8[9] & 1) != 0 && v31 )
  {
    v22 = (const __m128i *)sub_731370((__int64)a1, (__int64)v10, v11, v14, v12, v13);
    return sub_730620((__int64)a1, v22);
  }
  return result;
}
