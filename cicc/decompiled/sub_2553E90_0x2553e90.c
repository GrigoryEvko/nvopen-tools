// Function: sub_2553E90
// Address: 0x2553e90
//
bool __fastcall sub_2553E90(__int64 a1, unsigned __int64 a2)
{
  __int64 *v2; // r12
  unsigned __int8 *v3; // r13
  bool result; // al
  int v5; // ecx
  __int64 v6; // rax
  unsigned int v7; // edx
  __int64 *v8; // rbx
  __int64 *v9; // r15
  unsigned __int8 *v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int16 v15; // ax
  __int16 v16; // ax
  __int16 v17; // ax
  __int16 v18; // ax
  signed __int64 v19; // rax
  __int16 v20; // ax
  __int64 *v21; // rax
  __int16 v22; // ax
  __int16 v23; // ax
  __int64 *v24; // [rsp+8h] [rbp-B8h]
  __int64 *v25; // [rsp+10h] [rbp-B0h]
  __int64 *v26; // [rsp+20h] [rbp-A0h]
  bool v27; // [rsp+28h] [rbp-98h]
  __int64 v28; // [rsp+38h] [rbp-88h] BYREF
  __m128i v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v30; // [rsp+50h] [rbp-70h] BYREF
  __int64 v31; // [rsp+58h] [rbp-68h]
  _BYTE v32[16]; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v33; // [rsp+70h] [rbp-50h] BYREF
  __int64 v34; // [rsp+78h] [rbp-48h]
  _BYTE v35[64]; // [rsp+80h] [rbp-40h] BYREF

  v2 = (__int64 *)a2;
  v3 = (unsigned __int8 *)sub_250D070((_QWORD *)a2);
  if ( (unsigned int)((char)sub_2509800((_QWORD *)a2) - 6) > 1 )
    return *((_QWORD *)v3 + 2) == 0;
  result = 1;
  v5 = *v3;
  if ( (unsigned int)(v5 - 12) > 1 )
  {
    if ( (_BYTE)v5 != 20 )
      goto LABEL_9;
    v6 = *((_QWORD *)v3 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
      v6 = **(_QWORD **)(v6 + 16);
    v7 = *(_DWORD *)(v6 + 8);
    result = 1;
    if ( v7 >> 8 )
    {
LABEL_9:
      v30 = (__int64 *)v32;
      v31 = 0x100000000LL;
      LODWORD(v33) = 89;
      sub_2515D00(a1, (__m128i *)a2, (int *)&v33, 1, (__int64)&v30, 1);
      v8 = &v30[(unsigned int)v31];
      if ( v8 != v30 )
      {
        v9 = v30;
        while ( (unsigned __int16)sub_A71E10(v9) )
        {
          if ( v8 == ++v9 )
            goto LABEL_16;
        }
        goto LABEL_13;
      }
LABEL_16:
      if ( (unsigned __int8)sub_2509800((_QWORD *)a2) != 7 || (a2 = sub_250C680((__int64 *)a2)) == 0 )
      {
LABEL_17:
        v10 = sub_250CBE0(v2, a2);
        result = 0;
        if ( v10 )
        {
          LODWORD(v34) = 458752;
          v33 = (__int64 *)&unk_4A16F38;
          sub_2553CD0(v2, v10, (__int64)&v33);
          result = 0;
          if ( (v34 & 7) == 7 )
          {
            v11 = (__int64 *)sub_BD5C60((__int64)v3);
            v29.m128i_i64[0] = sub_A77AD0(v11, 0);
            sub_2516380(a1, v2, (__int64)&v29, 1, 0);
            result = 1;
          }
        }
        goto LABEL_14;
      }
      v33 = (__int64 *)v35;
      v34 = 0x100000000LL;
      v28 = 0x5100000059LL;
      sub_250D230((unsigned __int64 *)&v29, a2, 6, 0);
      sub_2515D00(a1, &v29, (int *)&v28, 2, (__int64)&v33, 1);
      v12 = v33;
      v13 = (unsigned int)v34;
      a2 = (unsigned __int64)&v33[v13];
      v14 = (v13 * 8) >> 5;
      v24 = (__int64 *)a2;
      if ( v14 )
      {
        v25 = &v33[4 * v14];
        while ( 1 )
        {
          v29.m128i_i64[0] = *v12;
          if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
            break;
          v15 = sub_A71E10(v29.m128i_i64);
          a2 = HIBYTE(v15);
          LOBYTE(a2) = v15 | HIBYTE(v15);
          if ( !v15 )
            break;
          v26 = v12 + 1;
          v29.m128i_i64[0] = v12[1];
          if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
            goto LABEL_41;
          v16 = sub_A71E10(v29.m128i_i64);
          a2 = HIBYTE(v16);
          LOBYTE(a2) = v16 | HIBYTE(v16);
          if ( !v16 )
            goto LABEL_41;
          v26 = v12 + 2;
          v29.m128i_i64[0] = v12[2];
          if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
            goto LABEL_41;
          v17 = sub_A71E10(v29.m128i_i64);
          a2 = HIBYTE(v17);
          LOBYTE(a2) = v17 | HIBYTE(v17);
          if ( !v17
            || (v26 = v12 + 3, v29.m128i_i64[0] = v12[3], (unsigned int)sub_A71AE0(v29.m128i_i64) == 81)
            || (v18 = sub_A71E10(v29.m128i_i64), a2 = HIBYTE(v18), LOBYTE(a2) = v18 | HIBYTE(v18), !v18) )
          {
LABEL_41:
            v12 = v26;
            break;
          }
          v12 += 4;
          if ( v25 == v12 )
            goto LABEL_35;
        }
LABEL_24:
        if ( v24 != v12 )
        {
          v21 = (__int64 *)sub_BD5C60((__int64)v3);
          v29.m128i_i64[0] = sub_A77AD0(v21, 0);
          sub_2516380(a1, v2, (__int64)&v29, 1, 0);
          if ( v33 != (__int64 *)v35 )
            _libc_free((unsigned __int64)v33);
LABEL_13:
          result = 1;
LABEL_14:
          if ( v30 != (__int64 *)v32 )
          {
            v27 = result;
            _libc_free((unsigned __int64)v30);
            return v27;
          }
          return result;
        }
        goto LABEL_25;
      }
LABEL_35:
      v19 = (char *)v24 - (char *)v12;
      if ( (char *)v24 - (char *)v12 != 16 )
      {
        if ( v19 != 24 )
        {
          if ( v19 != 8 )
            goto LABEL_25;
          goto LABEL_38;
        }
        v29.m128i_i64[0] = *v12;
        if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
          goto LABEL_24;
        v22 = sub_A71E10(v29.m128i_i64);
        a2 = HIBYTE(v22);
        LOBYTE(a2) = v22 | HIBYTE(v22);
        if ( !v22 )
          goto LABEL_24;
        ++v12;
      }
      v29.m128i_i64[0] = *v12;
      if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
        goto LABEL_24;
      v23 = sub_A71E10(v29.m128i_i64);
      a2 = HIBYTE(v23);
      LOBYTE(a2) = v23 | HIBYTE(v23);
      if ( !v23 )
        goto LABEL_24;
      ++v12;
LABEL_38:
      v29.m128i_i64[0] = *v12;
      if ( (unsigned int)sub_A71AE0(v29.m128i_i64) == 81 )
        goto LABEL_24;
      v20 = sub_A71E10(v29.m128i_i64);
      a2 = HIBYTE(v20);
      LOBYTE(a2) = v20 | HIBYTE(v20);
      if ( !v20 )
        goto LABEL_24;
LABEL_25:
      if ( v33 != (__int64 *)v35 )
        _libc_free((unsigned __int64)v33);
      goto LABEL_17;
    }
  }
  return result;
}
