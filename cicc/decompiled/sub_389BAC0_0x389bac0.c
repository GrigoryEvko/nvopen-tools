// Function: sub_389BAC0
// Address: 0x389bac0
//
char __fastcall sub_389BAC0(__int64 **a1, __int64 a2, unsigned int *a3, _QWORD *a4, __int64 *a5, char a6)
{
  unsigned int *v8; // rbx
  __int64 v9; // rcx
  __int64 v11; // rsi
  const char *v12; // rax
  unsigned __int64 v13; // rsi
  char result; // al
  __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rcx
  _QWORD *v21; // rax
  __m128i *v22; // rax
  __m128i *v23; // rax
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // rdi
  unsigned int v27; // eax
  char v28; // r13
  __int64 *v29; // rsi
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  __int64 *v33; // rsi
  __int64 i; // rax
  unsigned __int64 v35; // rsi
  char v36; // al
  __int16 *v37; // rax
  unsigned __int64 v38; // [rsp+0h] [rbp-C0h]
  char v39; // [rsp+Ch] [rbp-B4h]
  char v40; // [rsp+Ch] [rbp-B4h]
  char v41; // [rsp+Ch] [rbp-B4h]
  unsigned int v42; // [rsp+Ch] [rbp-B4h]
  unsigned __int64 **v43; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v44; // [rsp+20h] [rbp-A0h]
  __int64 v45[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v46; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v47; // [rsp+50h] [rbp-70h] BYREF
  __int64 v48; // [rsp+58h] [rbp-68h]
  __m128i v49; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 *v50; // [rsp+70h] [rbp-50h] BYREF
  const char *v51; // [rsp+78h] [rbp-48h]
  __m128i v52; // [rsp+80h] [rbp-40h] BYREF

  v8 = a3;
  v9 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v9 == 12 )
  {
    v52.m128i_i8[1] = 1;
    v12 = "functions are not values, refer to them as pointers";
LABEL_4:
    v13 = *((_QWORD *)v8 + 1);
    v50 = (unsigned __int64 *)v12;
    v52.m128i_i8[0] = 3;
    return sub_38814C0((__int64)(a1 + 1), v13, (__int64)&v50);
  }
  else
  {
    v11 = *a3;
    switch ( *a3 )
    {
      case 0u:
        if ( !a5 )
          goto LABEL_61;
        v16 = sub_389ACD0(a5, a3[4], a2, *((_QWORD *)a3 + 1), a6);
        *a4 = v16;
        return v16 == 0;
      case 1u:
        v17 = sub_38990C0(a1, a3[4], a2, *((_QWORD *)a3 + 1));
        *a4 = v17;
        return v17 == 0;
      case 2u:
        if ( !a5 )
        {
LABEL_61:
          v52.m128i_i8[1] = 1;
          v12 = "invalid use of function-local name";
          goto LABEL_4;
        }
        v18 = sub_389A230(a5, (__int64)(a3 + 8), a2, *((_QWORD *)a3 + 1), a6);
        *a4 = v18;
        return v18 == 0;
      case 3u:
        v19 = sub_38987A0(a1, (__int64)(a3 + 8), a2, *((_QWORD *)a3 + 1));
        *a4 = v19;
        return v19 == 0;
      case 4u:
        if ( (_BYTE)v9 != 11 )
        {
          v52.m128i_i8[1] = 1;
          v12 = "integer constant must have integer type";
          goto LABEL_4;
        }
        v27 = sub_1643030(a2);
        v28 = *((_BYTE *)v8 + 108);
        v29 = (__int64 *)(v8 + 24);
        if ( v28 )
          sub_16A5D10((__int64)&v50, (__int64)v29, v27);
        else
          sub_16A5D70((__int64)&v50, v29, v27);
        v30 = (unsigned int)v51;
        v31 = (unsigned __int64)v50;
        if ( v8[26] > 0x40 )
        {
          v32 = *((_QWORD *)v8 + 12);
          if ( v32 )
          {
            v38 = (unsigned __int64)v50;
            v42 = (unsigned int)v51;
            j_j___libc_free_0_0(v32);
            v31 = v38;
            v30 = v42;
          }
        }
        *((_QWORD *)v8 + 12) = v31;
        v8[26] = v30;
        *((_BYTE *)v8 + 108) = v28;
        *a4 = sub_159C0E0(*a1, (__int64)(v8 + 24));
        return 0;
      case 5u:
        v20 = (unsigned int)(v9 - 1);
        if ( (unsigned __int8)v20 > 5u || !(unsigned __int8)sub_1594800(a2, (__int64)(a3 + 28), (__int64)a3, v20) )
        {
          v52.m128i_i8[1] = 1;
          v12 = "floating point constant invalid for type";
          goto LABEL_4;
        }
        if ( *((void **)v8 + 15) != sub_1698280() )
          goto LABEL_20;
        v36 = *(_BYTE *)(a2 + 8);
        if ( v36 == 1 )
        {
          v37 = (__int16 *)sub_1698260();
        }
        else
        {
          if ( v36 != 2 )
            goto LABEL_20;
          v37 = (__int16 *)sub_1698270();
        }
        sub_16A3360((__int64)(v8 + 28), v37, 0, (bool *)&v50);
LABEL_20:
        v21 = (_QWORD *)sub_159CCF0(*a1, (__int64)(v8 + 28));
        *a4 = v21;
        if ( a2 == *v21 )
          return 0;
        sub_3888960(v45, a2);
        v22 = (__m128i *)sub_2241130(
                           (unsigned __int64 *)v45,
                           0,
                           0,
                           "floating point constant does not have type '",
                           0x2Cu);
        v47 = &v49;
        if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
        {
          v49 = _mm_loadu_si128(v22 + 1);
        }
        else
        {
          v47 = (__m128i *)v22->m128i_i64[0];
          v49.m128i_i64[0] = v22[1].m128i_i64[0];
        }
        v48 = v22->m128i_i64[1];
        v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
        v22->m128i_i64[1] = 0;
        v22[1].m128i_i8[0] = 0;
        if ( v48 == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v23 = (__m128i *)sub_2241490((unsigned __int64 *)&v47, "'", 1u);
        v50 = (unsigned __int64 *)&v52;
        if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
        {
          v52 = _mm_loadu_si128(v23 + 1);
        }
        else
        {
          v50 = (unsigned __int64 *)v23->m128i_i64[0];
          v52.m128i_i64[0] = v23[1].m128i_i64[0];
        }
        v51 = (const char *)v23->m128i_i64[1];
        v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
        v23->m128i_i64[1] = 0;
        v23[1].m128i_i8[0] = 0;
        v24 = *((_QWORD *)v8 + 1);
        v44 = 260;
        v43 = &v50;
        result = sub_38814C0((__int64)(a1 + 1), v24, (__int64)&v43);
        if ( v50 != (unsigned __int64 *)&v52 )
        {
          v39 = result;
          j_j___libc_free_0((unsigned __int64)v50);
          result = v39;
        }
        if ( v47 != &v49 )
        {
          v40 = result;
          j_j___libc_free_0((unsigned __int64)v47);
          result = v40;
        }
        if ( (__int64 *)v45[0] != &v46 )
        {
          v41 = result;
          j_j___libc_free_0(v45[0]);
          return v41;
        }
        return result;
      case 6u:
        if ( (_BYTE)v9 != 15 )
        {
          v52.m128i_i8[1] = 1;
          v12 = "null must be a pointer type";
          goto LABEL_4;
        }
        *a4 = sub_1599A20((__int64 **)a2);
        return 0;
      case 7u:
        if ( (unsigned __int8)v9 != 12 && (unsigned __int8)v9 != 0 && (_BYTE)v9 != 7 )
          goto LABEL_63;
        v52.m128i_i8[1] = 1;
        v12 = "invalid type for undef constant";
        goto LABEL_4;
      case 8u:
        LOBYTE(a3) = (_BYTE)v9 != 12;
        if ( (_BYTE)v9 != 0 && (_BYTE)v9 != 12 && (_BYTE)v9 != 7 )
          goto LABEL_53;
        v52.m128i_i8[1] = 1;
        v12 = "invalid type for null constant";
        goto LABEL_4;
      case 9u:
        if ( (_BYTE)v9 != 10 )
        {
          v52.m128i_i8[1] = 1;
          v12 = "invalid type for none constant";
          goto LABEL_4;
        }
LABEL_53:
        *a4 = sub_15A06D0((__int64 **)a2, v11, (__int64)a3, v9);
        return 0;
      case 0xAu:
        if ( (_BYTE)v9 != 14 || *(_QWORD *)(a2 + 32) )
        {
          v52.m128i_i8[1] = 1;
          v12 = "invalid empty array initializer";
          goto LABEL_4;
        }
LABEL_63:
        *a4 = sub_1599EF0((__int64 **)a2);
        return 0;
      case 0xBu:
        v25 = (_QWORD *)*((_QWORD *)a3 + 18);
        if ( a2 != *v25 )
          goto LABEL_46;
        goto LABEL_52;
      case 0xCu:
        v26 = *((_QWORD *)a3 + 3);
        if ( !v26 || !sub_15F1BE0(v26, *((_BYTE **)a3 + 8), *((_QWORD *)a3 + 9)) )
        {
          v52.m128i_i8[1] = 1;
          v12 = "invalid type for inline asm constraint string";
          goto LABEL_4;
        }
        *a4 = sub_15EE570(
                *((__int64 ***)v8 + 3),
                *((_BYTE **)v8 + 4),
                *((_QWORD *)v8 + 5),
                *((_BYTE **)v8 + 8),
                *((_QWORD *)v8 + 9),
                v8[4] & 1,
                (v8[4] & 2) != 0,
                v8[4] >> 2);
        return 0;
      case 0xDu:
      case 0xEu:
        if ( (_BYTE)v9 != 13 )
        {
LABEL_46:
          v52.m128i_i8[1] = 1;
          v12 = "constant expression type mismatch";
          goto LABEL_4;
        }
        v15 = a3[4];
        if ( (_DWORD)v15 != *(_DWORD *)(a2 + 12) )
        {
          v52.m128i_i8[1] = 1;
          v12 = "initializer with struct type has wrong # elements";
          goto LABEL_4;
        }
        LOBYTE(v9) = (_DWORD)v11 == 14;
        if ( ((_DWORD)v11 == 14) != ((*(_DWORD *)(a2 + 8) & 0x200) != 0) )
        {
          v52.m128i_i8[1] = 1;
          v12 = "packed'ness of initializer and type don't match";
          goto LABEL_4;
        }
        v33 = (__int64 *)*((_QWORD *)v8 + 19);
        if ( (_DWORD)v15 )
        {
          for ( i = 0; i != v15; ++i )
          {
            v9 = *(_QWORD *)v33[i];
            if ( *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * i) != v9 )
            {
              LODWORD(v45[0]) = i;
              v35 = *((_QWORD *)v8 + 1);
              v47 = (__m128i *)"element ";
              v52.m128i_i16[0] = 770;
              v48 = v45[0];
              v49.m128i_i16[0] = 2307;
              v50 = (unsigned __int64 *)&v47;
              v51 = " of struct initializer doesn't match struct element type";
              return sub_38814C0((__int64)(a1 + 1), v35, (__int64)&v50);
            }
          }
        }
        else
        {
          v15 = 0;
        }
        v25 = (_QWORD *)sub_159F090((__int64 **)a2, v33, v15, v9);
LABEL_52:
        *a4 = v25;
        result = 0;
        break;
    }
  }
  return result;
}
