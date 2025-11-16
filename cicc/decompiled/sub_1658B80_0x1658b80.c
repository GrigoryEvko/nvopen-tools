// Function: sub_1658B80
// Address: 0x1658b80
//
char __fastcall sub_1658B80(_BYTE *a1, __int64 a2, char a3, __int64 *a4)
{
  __int64 *v5; // rbx
  __int64 *v6; // rax
  __int64 *v7; // r12
  int v8; // kr00_4
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // r12
  _BYTE *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __m128i *v17; // rax
  __int64 v18; // rcx
  __m128i *v19; // rax
  __int64 *v22; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v23[2]; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v24; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD v26[2]; // [rsp+40h] [rbp-C0h] BYREF
  __m128i *v27; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-A8h]
  __m128i v29; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD v30[2]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD v31[2]; // [rsp+80h] [rbp-80h] BYREF
  __m128i *v32; // [rsp+90h] [rbp-70h] BYREF
  __int64 v33; // [rsp+98h] [rbp-68h]
  __m128i v34; // [rsp+A0h] [rbp-60h] BYREF
  _OWORD *v35; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v36; // [rsp+B8h] [rbp-48h]
  _OWORD v37[4]; // [rsp+C0h] [rbp-40h] BYREF

  v23[0] = a2;
  v22 = a4;
  v5 = (__int64 *)sub_155EE30(v23);
  v6 = (__int64 *)sub_155EE40(v23);
  if ( v5 == v6 )
    return (char)v6;
  v7 = v6;
  while ( 1 )
  {
    v24 = *v5;
    LOBYTE(v6) = sub_155D3E0((__int64)&v24);
    if ( !(_BYTE)v6 )
      break;
LABEL_6:
    if ( v7 == ++v5 )
      return (char)v6;
  }
  v8 = sub_155D410(&v24);
  LOBYTE(v6) = v8;
  switch ( v8 )
  {
    case 2:
    case 3:
    case 4:
    case 5:
    case 7:
    case 8:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 21:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 33:
    case 34:
    case 35:
    case 39:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 56:
      if ( a3 )
        goto LABEL_6;
      sub_155D8D0((__int64)v25, &v24, 0);
      v17 = (__m128i *)sub_2241130(v25, 0, 0, "Attribute '", 11);
      v27 = &v29;
      if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
      {
        v29 = _mm_loadu_si128(v17 + 1);
      }
      else
      {
        v27 = (__m128i *)v17->m128i_i64[0];
        v29.m128i_i64[0] = v17[1].m128i_i64[0];
      }
      v18 = v17->m128i_i64[1];
      v28 = v18;
      v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
      v17->m128i_i64[1] = 0;
      v17[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v28) <= 0x1B )
        goto LABEL_49;
      v19 = (__m128i *)sub_2241490(&v27, "' only applies to functions!", 28, v18);
      v35 = v37;
      if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
      {
        v37[0] = _mm_loadu_si128(v19 + 1);
      }
      else
      {
        v35 = (_OWORD *)v19->m128i_i64[0];
        *(_QWORD *)&v37[0] = v19[1].m128i_i64[0];
      }
      v36 = v19->m128i_i64[1];
      v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
      v19->m128i_i64[1] = 0;
      v19[1].m128i_i8[0] = 0;
      v34.m128i_i16[0] = 260;
      v32 = (__m128i *)&v35;
      sub_1658A90(a1, (__int64)&v32, (__int64 *)&v22);
      if ( v35 != v37 )
        j_j___libc_free_0(v35, *(_QWORD *)&v37[0] + 1LL);
      if ( v27 != &v29 )
        j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
      v6 = v26;
      if ( (_QWORD *)v25[0] != v26 )
        LOBYTE(v6) = j_j___libc_free_0(v25[0], v26[0] + 1LL);
      return (char)v6;
    default:
      LOBYTE(v6) = v8;
      if ( !a3 )
        goto LABEL_6;
      LODWORD(v6) = sub_155D410(&v24);
      if ( (unsigned int)((_DWORD)v6 - 36) <= 1 || (_DWORD)v6 == 57 )
        goto LABEL_6;
      sub_155D8D0((__int64)v30, &v24, 0);
      v9 = (__m128i *)sub_2241130(v30, 0, 0, "Attribute '", 11);
      v32 = &v34;
      if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
      {
        v34 = _mm_loadu_si128(v9 + 1);
      }
      else
      {
        v32 = (__m128i *)v9->m128i_i64[0];
        v34.m128i_i64[0] = v9[1].m128i_i64[0];
      }
      v10 = v9->m128i_i64[1];
      v33 = v10;
      v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
      v9->m128i_i64[1] = 0;
      v9[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v33) <= 0x1D )
LABEL_49:
        sub_4262D8((__int64)"basic_string::append");
      v11 = (__m128i *)sub_2241490(&v32, "' does not apply to functions!", 30, v10);
      v35 = v37;
      if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
      {
        v37[0] = _mm_loadu_si128(v11 + 1);
      }
      else
      {
        v35 = (_OWORD *)v11->m128i_i64[0];
        *(_QWORD *)&v37[0] = v11[1].m128i_i64[0];
      }
      v36 = v11->m128i_i64[1];
      v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
      v11->m128i_i64[1] = 0;
      v11[1].m128i_i8[0] = 0;
      v29.m128i_i16[0] = 260;
      v27 = (__m128i *)&v35;
      v12 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
      {
        a1[72] = 1;
        goto LABEL_24;
      }
      sub_16E2CE0(&v27, *(_QWORD *)a1);
      v13 = *(_BYTE **)(v12 + 24);
      if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
      {
        sub_16E7DE0(v12, 10);
      }
      else
      {
        *(_QWORD *)(v12 + 24) = v13 + 1;
        *v13 = 10;
      }
      v14 = *(_QWORD *)a1;
      a1[72] = 1;
      if ( !v14 || !v22 )
        goto LABEL_24;
      if ( *((_BYTE *)v22 + 16) <= 0x17u )
      {
        sub_1553920(v22, v14, 1, (__int64)(a1 + 16));
        v15 = *(_QWORD *)a1;
        v16 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          goto LABEL_23;
      }
      else
      {
        sub_155BD40((__int64)v22, v14, (__int64)(a1 + 16), 0);
        v15 = *(_QWORD *)a1;
        v16 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
LABEL_23:
          *(_QWORD *)(v15 + 24) = v16 + 1;
          *v16 = 10;
LABEL_24:
          if ( v35 != v37 )
            j_j___libc_free_0(v35, *(_QWORD *)&v37[0] + 1LL);
          if ( v32 != &v34 )
            j_j___libc_free_0(v32, v34.m128i_i64[0] + 1);
          v6 = v31;
          if ( (_QWORD *)v30[0] != v31 )
            LOBYTE(v6) = j_j___libc_free_0(v30[0], v31[0] + 1LL);
          return (char)v6;
        }
      }
      sub_16E7DE0(v15, 10);
      goto LABEL_24;
  }
}
