// Function: sub_16595D0
// Address: 0x16595d0
//
void __fastcall sub_16595D0(_BYTE *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r13
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __m128i *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  __int64 v15; // r14
  char v16; // al
  char v17; // al
  char v18; // al
  char v19; // al
  const char *v20; // rax
  __int64 v21; // rax
  int v22; // [rsp+0h] [rbp-F0h]
  int v23; // [rsp+8h] [rbp-E8h]
  char v24; // [rsp+13h] [rbp-DDh]
  char v25; // [rsp+14h] [rbp-DCh]
  char v26; // [rsp+15h] [rbp-DBh]
  char v27; // [rsp+16h] [rbp-DAh]
  char v28; // [rsp+17h] [rbp-D9h]
  __int64 *v29; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v30; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v32; // [rsp+38h] [rbp-B8h] BYREF
  __int64 **v33; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v34; // [rsp+50h] [rbp-A0h]
  _BYTE v35[32]; // [rsp+60h] [rbp-90h] BYREF
  __m128i *v36; // [rsp+80h] [rbp-70h] BYREF
  __int64 v37; // [rsp+88h] [rbp-68h]
  __m128i v38; // [rsp+90h] [rbp-60h] BYREF
  __m128i *v39; // [rsp+A0h] [rbp-50h] BYREF
  __int64 **v40; // [rsp+A8h] [rbp-48h]
  __m128i v41; // [rsp+B0h] [rbp-40h] BYREF

  v31 = a2;
  v30 = a3;
  v29 = a4;
  if ( !a3 )
    return;
  v32 = sub_1560240(&v30);
  if ( (unsigned __int8)sub_155EE10((__int64)&v32, 6)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 19)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 53)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 22)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 38)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 11)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 55)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 54) )
  {
    v5 = *(_QWORD *)a1;
    v39 = (__m128i *)"Attributes 'byval', 'inalloca', 'nest', 'sret', 'nocapture', 'returned', 'swiftself', and 'swifterr"
                     "or' do not apply to return values!";
    v41.m128i_i16[0] = 259;
    if ( !v5 )
    {
      a1[72] = 1;
      return;
    }
    sub_16E2CE0(&v39, v5);
    v6 = *(_BYTE **)(v5 + 24);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
    {
      sub_16E7DE0(v5, 10);
    }
    else
    {
      *(_QWORD *)(v5 + 24) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_QWORD *)a1;
    a1[72] = 1;
    if ( !v7 || !v29 )
      return;
    if ( *((_BYTE *)v29 + 16) > 0x17u )
    {
      sub_155BD40((__int64)v29, v7, (__int64)(a1 + 16), 0);
      v8 = *(_QWORD *)a1;
      v9 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v9 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_12;
    }
    else
    {
      sub_1553920(v29, v7, 1, (__int64)(a1 + 16));
      v8 = *(_QWORD *)a1;
      v9 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v9 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
LABEL_12:
        *(_QWORD *)(v8 + 24) = v9 + 1;
        *v9 = 10;
        return;
      }
    }
    sub_16E7DE0(v8, 10);
    return;
  }
  if ( (unsigned __int8)sub_155EE10((__int64)&v32, 37)
    || (unsigned __int8)sub_155EE10((__int64)&v32, 57)
    || (v28 = sub_155EE10((__int64)&v32, 36)) != 0 )
  {
    sub_155F820((__int64)v35, &v32, 0);
    v10 = (__m128i *)sub_2241130(v35, 0, 0, "Attribute '", 11);
    v36 = &v38;
    if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
    {
      v38 = _mm_loadu_si128(v10 + 1);
    }
    else
    {
      v36 = (__m128i *)v10->m128i_i64[0];
      v38.m128i_i64[0] = v10[1].m128i_i64[0];
    }
    v11 = v10->m128i_i64[1];
    v37 = v11;
    v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
    v10->m128i_i64[1] = 0;
    v10[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v37) <= 0x23 )
      sub_4262D8((__int64)"basic_string::append");
    v12 = sub_2241490(&v36, "' does not apply to function returns", 36, v11);
    v39 = &v41;
    if ( *(_QWORD *)v12 == v12 + 16 )
    {
      v41 = _mm_loadu_si128((const __m128i *)(v12 + 16));
    }
    else
    {
      v39 = *(__m128i **)v12;
      v41.m128i_i64[0] = *(_QWORD *)(v12 + 16);
    }
    v40 = *(__int64 ***)(v12 + 8);
    *(_QWORD *)v12 = v12 + 16;
    *(_QWORD *)(v12 + 8) = 0;
    *(_BYTE *)(v12 + 16) = 0;
    v34 = 260;
    v33 = (__int64 **)&v39;
    sub_1658A90(a1, (__int64)&v33, (__int64 *)&v29);
    sub_2240A30(&v39);
    sub_2240A30(&v36);
    sub_2240A30(v35);
    return;
  }
  sub_1659040((__int64)a1, v32, **(_QWORD **)(v31 + 16), v29);
  v13 = v31;
  v23 = *(_DWORD *)(v31 + 12) - 1;
  if ( *(_DWORD *)(v31 + 12) != 1 )
  {
    v24 = 0;
    v14 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    while ( 1 )
    {
      v22 = v14 + 1;
      v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * (v14 + 1));
      v36 = (__m128i *)sub_1560230(&v30, v14);
      sub_1659040((__int64)a1, (__int64)v36, v15, v29);
      v16 = sub_155EE10((__int64)&v36, 19);
      if ( v16 )
      {
        if ( v28 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "More than one parameter has attribute nest!";
          goto LABEL_57;
        }
        v28 = v16;
      }
      if ( (unsigned __int8)sub_155EE10((__int64)&v36, 38) )
      {
        if ( v27 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "More than one parameter has attribute returned!";
          goto LABEL_57;
        }
        v27 = sub_16430A0(v15, **(_QWORD **)(v31 + 16));
        if ( !v27 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Incompatible argument and return types for 'returned' attribute";
          goto LABEL_57;
        }
      }
      v17 = sub_155EE10((__int64)&v36, 53);
      if ( v17 )
      {
        if ( v26 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Cannot have multiple 'sret' parameters!";
          goto LABEL_57;
        }
        if ( v14 > 1 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Attribute 'sret' is not on first or second parameter!";
          goto LABEL_57;
        }
        v26 = v17;
      }
      v18 = sub_155EE10((__int64)&v36, 55);
      if ( v18 )
      {
        if ( v25 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Cannot have multiple 'swiftself' parameters!";
          goto LABEL_57;
        }
        v25 = v18;
      }
      v19 = sub_155EE10((__int64)&v36, 54);
      if ( v19 )
      {
        if ( v24 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Cannot have multiple 'swifterror' parameters!";
          goto LABEL_57;
        }
        v24 = v19;
      }
      if ( (unsigned __int8)sub_155EE10((__int64)&v36, 11) && *(_DWORD *)(v31 + 12) - 2 != (_DWORD)v14 )
      {
        v41.m128i_i8[1] = 1;
        v20 = "inalloca isn't on the last parameter!";
        goto LABEL_57;
      }
      ++v14;
      if ( v23 == v22 )
        break;
      v13 = v31;
    }
  }
  if ( sub_15602F0(&v30, -1) )
  {
    v21 = sub_1560250(&v30);
    sub_1658B80(a1, v21, 1, v29);
    if ( (unsigned __int8)sub_1560180((__int64)&v30, 36) && (unsigned __int8)sub_1560180((__int64)&v30, 37) )
    {
      v41.m128i_i8[1] = 1;
      v20 = "Attributes 'readnone and readonly' are incompatible!";
    }
    else if ( (unsigned __int8)sub_1560180((__int64)&v30, 36) && (unsigned __int8)sub_1560180((__int64)&v30, 57) )
    {
      v41.m128i_i8[1] = 1;
      v20 = "Attributes 'readnone and writeonly' are incompatible!";
    }
    else
    {
      if ( !(unsigned __int8)sub_1560180((__int64)&v30, 37) || !(unsigned __int8)sub_1560180((__int64)&v30, 57) )
      {
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 36) && (unsigned __int8)sub_1560180((__int64)&v30, 14) )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Attributes 'readnone and inaccessiblemem_or_argmemonly' are incompatible!";
          goto LABEL_57;
        }
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 36) && (unsigned __int8)sub_1560180((__int64)&v30, 13) )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Attributes 'readnone and inaccessiblememonly' are incompatible!";
          goto LABEL_57;
        }
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 26) && (unsigned __int8)sub_1560180((__int64)&v30, 3) )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Attributes 'noinline and alwaysinline' are incompatible!";
          goto LABEL_57;
        }
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 35) )
        {
          if ( !(unsigned __int8)sub_1560180((__int64)&v30, 26) )
          {
            v41.m128i_i8[1] = 1;
            v20 = "Attribute 'optnone' requires 'noinline'!";
            goto LABEL_57;
          }
          if ( (unsigned __int8)sub_1560180((__int64)&v30, 34) )
          {
            v41.m128i_i8[1] = 1;
            v20 = "Attributes 'optsize and optnone' are incompatible!";
            goto LABEL_57;
          }
          if ( (unsigned __int8)sub_1560180((__int64)&v30, 17) )
          {
            v41.m128i_i8[1] = 1;
            v20 = "Attributes 'minsize and optnone' are incompatible!";
            goto LABEL_57;
          }
        }
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 16) && *((_BYTE *)v29 + 32) >> 6 != 2 )
        {
          v41.m128i_i8[1] = 1;
          v20 = "Attribute 'jumptable' requires 'unnamed_addr'";
          goto LABEL_57;
        }
        if ( (unsigned __int8)sub_1560180((__int64)&v30, 2) )
        {
          sub_1560420((__int64)&v36, &v30, -1);
          v41.m128i_i64[0] = (__int64)a1;
          v39 = (__m128i *)&v31;
          v40 = &v29;
          if ( (unsigned __int8)sub_164FD80((__int64 **)&v39, (__int64)"element size", 12, (unsigned int)v36) )
          {
            if ( (_BYTE)v37 )
              sub_164FD80((__int64 **)&v39, (__int64)"number of elements", 18, HIDWORD(v36));
          }
        }
        return;
      }
      v41.m128i_i8[1] = 1;
      v20 = "Attributes 'readonly and writeonly' are incompatible!";
    }
LABEL_57:
    v39 = (__m128i *)v20;
    v41.m128i_i8[0] = 3;
    sub_1658A90(a1, (__int64)&v39, (__int64 *)&v29);
  }
}
