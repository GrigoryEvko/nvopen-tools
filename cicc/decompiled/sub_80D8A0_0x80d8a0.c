// Function: sub_80D8A0
// Address: 0x80d8a0
//
__int64 __fastcall sub_80D8A0(const __m128i *a1, unsigned int a2, __int64 a3, _QWORD *a4)
{
  const __m128i *v6; // r12
  __int64 result; // rax
  __int64 v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 i; // rax
  const char *v12; // r12
  size_t v13; // rax
  _QWORD *v14; // rdi
  __int8 v15; // al
  __int64 v16; // rdi
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  unsigned __int8 v22; // bl
  _QWORD *v23; // rdi
  __int64 v24; // rdx
  char *v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rbx
  __int64 *v28; // rax
  __int64 *v29; // r14
  __int64 *v30; // rax
  __int64 v31; // r11
  int v32; // edx
  __int8 v33; // al
  __int64 v34; // r9
  __int8 v35; // bl
  char *v36; // rax
  char v37; // r8
  bool v38; // zf
  __int64 v39; // rdx
  char v40[4]; // [rsp+0h] [rbp-D0h] BYREF
  int v41; // [rsp+4h] [rbp-CCh]
  __int16 v42; // [rsp+3Ah] [rbp-96h]
  char v43[4]; // [rsp+40h] [rbp-90h] BYREF
  int v44; // [rsp+44h] [rbp-8Ch]
  __int64 v45; // [rsp+48h] [rbp-88h] BYREF
  __int8 v46; // [rsp+50h] [rbp-80h]
  __int64 v47; // [rsp+58h] [rbp-78h]
  char *v48; // [rsp+60h] [rbp-70h]
  char *s; // [rsp+80h] [rbp-50h]
  __int64 v50; // [rsp+88h] [rbp-48h]
  __int64 v51; // [rsp+90h] [rbp-40h]

  v6 = a1;
  result = a1[10].m128i_u8[13];
LABEL_2:
  if ( (_BYTE)result == 12 )
  {
    if ( !v6[9].m128i_i64[0] )
    {
LABEL_44:
      v22 = v6[11].m128i_u8[0];
      switch ( v22 )
      {
        case 0u:
          return sub_812B60(&v6[11].m128i_u64[1], 0, a4);
        case 1u:
          v30 = sub_72E9A0((__int64)v6);
          return sub_816460(v30, 1, 0, a4);
        case 2u:
          return sub_8156E0((int)v6);
        case 3u:
          v31 = 0;
          v32 = 0;
          goto LABEL_71;
        case 4u:
          return sub_8156E0(v6[11].m128i_i64[1]);
        case 5u:
        case 7u:
        case 8u:
        case 9u:
        case 0xAu:
          v26 = sub_72F1F0((__int64)v6);
          return sub_818B40(v6[11].m128i_i64[1], v26, v22, a4);
        case 0xBu:
          v31 = v6[12].m128i_i64[0];
          v6 = (const __m128i *)v6[11].m128i_i64[1];
          v32 = 1;
LABEL_71:
          v33 = v6[5].m128i_i8[9];
          v34 = v6[11].m128i_i64[1];
          v35 = v6[12].m128i_i8[8];
          if ( (v33 & 0x40) != 0 )
          {
            v36 = 0;
            v37 = 3;
            if ( !v34 )
              v37 = v35 != 0 ? 5 : 0;
          }
          else
          {
            if ( (v33 & 8) != 0 )
              v36 = (char *)v6[1].m128i_i64[1];
            else
              v36 = (char *)v6->m128i_i64[1];
            v37 = 3;
            if ( !v34 )
              v37 = v35 != 0 ? 5 : 0;
            if ( v36 )
            {
              if ( !memcmp(v36, "operator \"\"", 0xBu) )
              {
                v36 += 11;
                v37 = 4;
              }
              else
              {
                v36 = 0;
              }
            }
          }
          v38 = v32 == 0;
          v39 = 0;
          if ( !v38 )
            v39 = v31;
          v46 = v6[12].m128i_i8[8];
          v43[0] = v37;
          v47 = v39;
          v45 = v34;
          v48 = v36;
          return sub_8156E0((int)v6);
        case 0xCu:
          *a4 += 24LL;
          sub_8238B0(qword_4F18BE0, "spclL_Z14__integer_packE", 24);
          sub_80D8A0(v6[11].m128i_i64[1], 1, 0, a4);
          goto LABEL_16;
        default:
          goto LABEL_10;
      }
    }
    a2 = 1;
    if ( !sub_72AE00((__int64)v6) )
      goto LABEL_7;
  }
  else
  {
    if ( !a2 || !v6[9].m128i_i64[0] )
      goto LABEL_4;
    if ( !sub_72AE00((__int64)v6) )
    {
LABEL_7:
      v8 = v6[9].m128i_i64[0];
      if ( *(_BYTE *)(v8 + 24) != 3 || dword_4D0425C )
        return sub_816460(v8, a2, 0, a4);
    }
  }
  result = v6[10].m128i_u8[13];
  if ( (unsigned __int8)result > 0xFu )
LABEL_10:
    sub_721090();
LABEL_4:
  while ( 2 )
  {
    switch ( (char)result )
    {
      case 0:
        v23 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        result = v23[2];
        if ( (unsigned __int64)(result + 1) > v23[1] )
        {
          sub_823810(v23);
          v23 = (_QWORD *)qword_4F18BE0;
          result = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v23[4] + result) = 63;
        ++v23[2];
        return result;
      case 1:
        if ( dword_4D0425C && a2 && v6->m128i_i64[1] && unk_4D04250 <= 0x75F8u )
          return sub_8156E0((int)v6);
        v25 = sub_6228C0(v6);
        v41 = 0;
        s = (char *)&v45;
        v42 = 0;
        v50 = 0;
        v51 = 0;
        v44 = 1;
        sub_80D480((__int64)v43, (__int64)v40, v25);
        result = sub_80F9E0(s);
        if ( s != (char *)&v45 )
          return sub_823A00(s, v50);
        return result;
      case 2:
        v20 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        v21 = v20[2];
        if ( (unsigned __int64)(v21 + 1) > v20[1] )
        {
          sub_823810(v20);
          v20 = (_QWORD *)qword_4F18BE0;
          v21 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v20[4] + v21) = 76;
        ++v20[2];
        sub_80F5E0(v6[8].m128i_i64[0], 0, a4);
        goto LABEL_16;
      case 3:
        v9 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        v10 = v9[2];
        if ( (unsigned __int64)(v10 + 1) > v9[1] )
        {
          sub_823810(v9);
          v9 = (_QWORD *)qword_4F18BE0;
          v10 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v9[4] + v10) = 76;
        ++v9[2];
        sub_80F5E0(v6[8].m128i_i64[0], 0, a4);
        for ( i = v6[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v12 = sub_70B4A0(*(_BYTE *)(i + 160), (__int64)v6[11].m128i_i64);
        v13 = strlen(v12);
        *a4 += v13;
        sub_8238B0(qword_4F18BE0, v12, v13);
LABEL_16:
        v14 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        result = v14[2];
        if ( (unsigned __int64)(result + 1) <= v14[1] )
          goto LABEL_17;
        goto LABEL_33;
      case 4:
        return sub_80F7E0(v6, a4);
      case 6:
        v15 = v6[11].m128i_i8[0];
        if ( v15 == 2 )
        {
          result = *(unsigned __int8 *)(v6[11].m128i_i64[1] + 173);
          if ( (_BYTE)result != 2 )
            goto LABEL_58;
          v6 = (const __m128i *)v6[11].m128i_i64[1];
          a2 = 0;
          continue;
        }
        if ( v15 == 3 )
        {
          v6 = (const __m128i *)v6[11].m128i_i64[1];
          a2 = 0;
          result = v6[10].m128i_u8[13];
          goto LABEL_2;
        }
LABEL_58:
        v27 = (__int64 *)v6[12].m128i_i64[1];
        v28 = 0;
        if ( !v27 )
          return sub_814FD0(v6, a4);
        while ( 1 )
        {
          v29 = (__int64 *)*v27;
          *v27 = (__int64)v28;
          v28 = v27;
          if ( !v29 )
            break;
          v27 = v29;
        }
        sub_815200(v6, v27, a4);
        while ( 1 )
        {
          result = *v27;
          *v27 = (__int64)v29;
          v29 = v27;
          if ( !result )
            break;
          v27 = (__int64 *)result;
        }
        return result;
      case 7:
        v16 = v6[12].m128i_i64[1];
        if ( (v6[12].m128i_i8[0] & 2) != 0 )
        {
          if ( !v16 )
          {
LABEL_28:
            v17 = (_QWORD *)qword_4F18BE0;
            ++*a4;
            v18 = v17[2];
            if ( (unsigned __int64)(v18 + 1) > v17[1] )
            {
              sub_823810(v17);
              v17 = (_QWORD *)qword_4F18BE0;
              v18 = *(_QWORD *)(qword_4F18BE0 + 16);
            }
            *(_BYTE *)(v17[4] + v18) = 76;
            ++v17[2];
            sub_80F5E0(v6[8].m128i_i64[0], 0, a4);
            v14 = (_QWORD *)qword_4F18BE0;
            ++*a4;
            v19 = v14[2];
            if ( (unsigned __int64)(v19 + 1) > v14[1] )
            {
              sub_823810(v14);
              v14 = (_QWORD *)qword_4F18BE0;
              v19 = *(_QWORD *)(qword_4F18BE0 + 16);
            }
            *(_BYTE *)(v14[4] + v19) = 48;
            ++v14[2];
            ++*a4;
            result = v14[2];
            if ( (unsigned __int64)(result + 1) > v14[1] )
            {
LABEL_33:
              sub_823810(v14);
              v14 = (_QWORD *)qword_4F18BE0;
              result = *(_QWORD *)(qword_4F18BE0 + 16);
            }
LABEL_17:
            *(_BYTE *)(v14[4] + result) = 69;
            ++v14[2];
            return result;
          }
        }
        else if ( !v16 )
        {
          goto LABEL_28;
        }
        return sub_8156E0(v16);
      case 9:
        return sub_817A40(v6[11].m128i_i64[0], v6[8].m128i_i64[0], 0, a4);
      case 10:
        v24 = v6[8].m128i_i64[0];
        if ( *(_BYTE *)(v24 + 140) == 5 )
          return sub_80F7E0(v6, a4);
        else
          return sub_817960(0, v6, v24, a4);
      case 11:
        return result;
      case 12:
        goto LABEL_44;
      case 15:
        switch ( v6[11].m128i_i8[0] )
        {
          case 2:
          case 7:
          case 8:
          case 0xB:
            return sub_8156E0(v6[11].m128i_i64[1]);
          case 6:
            return sub_80F5E0(v6[11].m128i_i64[1], 0, a4);
          case 0xD:
            return sub_816460(v6[11].m128i_i64[1], 0, 0, a4);
          default:
            goto LABEL_10;
        }
      default:
        goto LABEL_10;
    }
  }
}
