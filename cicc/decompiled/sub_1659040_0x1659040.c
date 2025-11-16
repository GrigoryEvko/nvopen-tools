// Function: sub_1659040
// Address: 0x1659040
//
void __fastcall sub_1659040(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  int v6; // ebx
  int v7; // ebx
  char v8; // r8
  int v9; // eax
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  const char *v12; // rax
  __int64 v13; // rdx
  const char *v14; // rax
  __m128i *v15; // rax
  const char *v16; // rax
  char v17; // [rsp+Fh] [rbp-151h]
  __int64 *v18; // [rsp+10h] [rbp-150h] BYREF
  __int64 v19[2]; // [rsp+18h] [rbp-148h] BYREF
  __int64 v20; // [rsp+28h] [rbp-138h] BYREF
  __m128i *v21; // [rsp+30h] [rbp-130h] BYREF
  __int16 v22; // [rsp+40h] [rbp-120h]
  _QWORD v23[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v24; // [rsp+60h] [rbp-100h] BYREF
  _QWORD v25[12]; // [rsp+70h] [rbp-F0h] BYREF
  __m128i v26; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v27; // [rsp+E0h] [rbp-80h] BYREF
  int v28; // [rsp+F0h] [rbp-70h]
  _BYTE v29[104]; // [rsp+F8h] [rbp-68h] BYREF

  v19[0] = a2;
  v18 = a4;
  if ( a2 )
  {
    sub_1658B80((_BYTE *)a1, a2, 0, a4);
    v6 = (unsigned __int8)sub_155EE10((__int64)v19, 6);
    v7 = (unsigned __int8)sub_155EE10((__int64)v19, 11) + v6;
    v8 = sub_155EE10((__int64)v19, 53);
    v9 = 1;
    if ( !v8 )
      v9 = (unsigned __int8)sub_155EE10((__int64)v19, 12);
    if ( (unsigned int)(unsigned __int8)sub_155EE10((__int64)v19, 19) + v9 + v7 > 1 )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'byval', 'inalloca', 'inreg', 'nest', and 'sret' are incompatible!";
LABEL_29:
      v27.m128i_i8[0] = 3;
      v26.m128i_i64[0] = (__int64)v12;
      sub_1658A90((_BYTE *)a1, (__int64)&v26, (__int64 *)&v18);
      return;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 11) && (unsigned __int8)sub_155EE10((__int64)v19, 37) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'inalloca and readonly' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 53) && (unsigned __int8)sub_155EE10((__int64)v19, 38) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'sret and returned' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 58) && (unsigned __int8)sub_155EE10((__int64)v19, 40) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'zeroext and signext' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 36) && (unsigned __int8)sub_155EE10((__int64)v19, 37) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'readnone and readonly' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 36) && (unsigned __int8)sub_155EE10((__int64)v19, 57) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'readnone and writeonly' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 37) && (unsigned __int8)sub_155EE10((__int64)v19, 57) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'readonly and writeonly' are incompatible!";
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 26) && (unsigned __int8)sub_155EE10((__int64)v19, 3) )
    {
      v27.m128i_i8[1] = 1;
      v12 = "Attributes 'noinline and alwaysinline' are incompatible!";
      goto LABEL_29;
    }
    sub_1560E30((__int64)v25, a3);
    sub_1563030(&v26, v19[0]);
    v17 = sub_1561CE0(&v26, v25);
    sub_164EC70((_QWORD *)v27.m128i_i64[1]);
    if ( v17 )
    {
      v20 = sub_1560BF0(*(__int64 **)(a1 + 64), v25);
      sub_155F820((__int64)v23, &v20, 0);
      v15 = (__m128i *)sub_2241130(v23, 0, 0, "Wrong types for attribute: ", 27);
      v26.m128i_i64[0] = (__int64)&v27;
      if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
      {
        v27 = _mm_loadu_si128(v15 + 1);
      }
      else
      {
        v26.m128i_i64[0] = v15->m128i_i64[0];
        v27.m128i_i64[0] = v15[1].m128i_i64[0];
      }
      v26.m128i_i64[1] = v15->m128i_i64[1];
      v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
      v15->m128i_i64[1] = 0;
      v15[1].m128i_i8[0] = 0;
      v22 = 260;
      v21 = &v26;
      sub_1658A90((_BYTE *)a1, (__int64)&v21, (__int64 *)&v18);
      if ( (__m128i *)v26.m128i_i64[0] != &v27 )
        j_j___libc_free_0(v26.m128i_i64[0], v27.m128i_i64[0] + 1);
      if ( (__int64 *)v23[0] != &v24 )
        j_j___libc_free_0(v23[0], v24 + 1);
      goto LABEL_35;
    }
    if ( *(_BYTE *)(a3 + 8) == 15 )
    {
      v10 = *(_QWORD *)(a3 + 24);
      v26.m128i_i64[0] = 0;
      v26.m128i_i64[1] = (__int64)v29;
      v27.m128i_i64[0] = (__int64)v29;
      v27.m128i_i64[1] = 4;
      v28 = 0;
      v11 = *(unsigned __int8 *)(v10 + 8);
      if ( (unsigned __int8)v11 > 0xFu || (v13 = 35454, !_bittest64(&v13, v11)) )
      {
        if ( ((unsigned int)(v11 - 13) > 1 && (_DWORD)v11 != 16 || !sub_16435F0(v10, (__int64)&v26))
          && ((unsigned __int8)sub_155EE10((__int64)v19, 6) || (unsigned __int8)sub_155EE10((__int64)v19, 11)) )
        {
          BYTE1(v24) = 1;
          v16 = "Attributes 'byval' and 'inalloca' do not support unsized types!";
          goto LABEL_56;
        }
        if ( *(_BYTE *)(*(_QWORD *)(a3 + 24) + 8LL) == 15 )
          goto LABEL_39;
LABEL_38:
        if ( !(unsigned __int8)sub_155EE10((__int64)v19, 54) )
          goto LABEL_39;
        BYTE1(v24) = 1;
        v16 = "Attribute 'swifterror' only applies to parameters with pointer to pointer type!";
LABEL_56:
        v23[0] = v16;
        LOBYTE(v24) = 3;
        sub_1658A90((_BYTE *)a1, (__int64)v23, (__int64 *)&v18);
LABEL_39:
        if ( v27.m128i_i64[0] != v26.m128i_i64[1] )
          _libc_free(v27.m128i_u64[0]);
        goto LABEL_35;
      }
      if ( (_BYTE)v11 != 15 )
        goto LABEL_38;
LABEL_35:
      sub_164EC70((_QWORD *)v25[3]);
      return;
    }
    if ( (unsigned __int8)sub_155EE10((__int64)v19, 6) )
    {
      v27.m128i_i8[1] = 1;
      v14 = "Attribute 'byval' only applies to parameters with pointer type!";
    }
    else
    {
      if ( !(unsigned __int8)sub_155EE10((__int64)v19, 54) )
        goto LABEL_35;
      v27.m128i_i8[1] = 1;
      v14 = "Attribute 'swifterror' only applies to parameters with pointer type!";
    }
    v26.m128i_i64[0] = (__int64)v14;
    v27.m128i_i8[0] = 3;
    sub_1658A90((_BYTE *)a1, (__int64)&v26, (__int64 *)&v18);
    goto LABEL_35;
  }
}
