// Function: sub_139D7E0
// Address: 0x139d7e0
//
__int64 __fastcall sub_139D7E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // r15
  unsigned int v7; // ebx
  __m128i *v8; // rdx
  unsigned __int64 v9; // rax
  __m128i si128; // xmm0
  __int64 v11; // rdi
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __m128i v14; // xmm0
  __int64 v15; // r12
  _BYTE *v16; // rax
  __int64 v17; // r14
  __int64 v18; // r11
  __int64 v19; // rax
  char *v20; // rbx
  char *v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r12
  _BYTE *v25; // rcx
  int v26; // edi
  _BYTE *v27; // r8
  __int64 v28; // rax
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  unsigned __int64 v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+38h] [rbp-68h]
  _BYTE *v38; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+48h] [rbp-58h]
  _BYTE v40[80]; // [rsp+50h] [rbp-50h] BYREF

  result = *(_QWORD *)(a1 + 160);
  if ( result )
  {
    v4 = *(_QWORD *)(result + 80);
    result += 72;
    v31 = result;
    v35 = v4;
    if ( v4 != result )
    {
      while ( 1 )
      {
        if ( !v35 )
          BUG();
        v5 = *(_QWORD *)(v35 + 24);
        v6 = a2;
        v37 = v35 + 16;
        if ( v35 + 16 != v5 )
          break;
LABEL_34:
        result = *(_QWORD *)(v35 + 8);
        v35 = result;
        if ( v31 == result )
          return result;
      }
      while ( 1 )
      {
        v17 = 0;
        if ( v5 )
          v17 = v5 - 24;
        v18 = *(_QWORD *)(a1 + 168);
        if ( dword_4F98B60 == 1 )
          break;
        if ( dword_4F98B60 != 2 )
        {
          v7 = sub_14A5450(v18, v17);
          goto LABEL_7;
        }
        v19 = 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
        {
          v20 = *(char **)(v17 - 8);
          v21 = &v20[v19];
        }
        else
        {
          v20 = (char *)(v17 - v19);
          v21 = (char *)v17;
        }
        v22 = v21 - v20;
        v39 = 0x400000000LL;
        v38 = v40;
        v23 = 0xAAAAAAAAAAAAAAABLL * (v22 >> 3);
        v24 = v23;
        if ( (unsigned __int64)v22 > 0x60 )
        {
          v32 = v22;
          v33 = v18;
          v34 = 0xAAAAAAAAAAAAAAABLL * (v22 >> 3);
          sub_16CD150(&v38, v40, v23, 8);
          v27 = v38;
          v26 = v39;
          LODWORD(v23) = v34;
          v18 = v33;
          v22 = v32;
          v25 = &v38[8 * (unsigned int)v39];
        }
        else
        {
          v25 = v40;
          v26 = 0;
          v27 = v40;
        }
        if ( v22 > 0 )
        {
          do
          {
            v28 = *(_QWORD *)v20;
            v25 += 8;
            v20 += 24;
            *((_QWORD *)v25 - 1) = v28;
            --v24;
          }
          while ( v24 );
          v27 = v38;
          v26 = v39;
        }
        LODWORD(v39) = v26 + v23;
        v7 = sub_14A5330(v18, v17, v27, (unsigned int)(v26 + v23));
        if ( v38 == v40 )
          goto LABEL_7;
        _libc_free((unsigned __int64)v38);
        v8 = *(__m128i **)(v6 + 24);
        v9 = *(_QWORD *)(v6 + 16) - (_QWORD)v8;
        if ( v7 != -1 )
        {
LABEL_8:
          if ( v9 <= 0x26 )
          {
            v11 = sub_16E7EE0(v6, "Cost Model: Found an estimated cost of ", 39);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_4289510);
            v8[2].m128i_i8[6] = 32;
            v11 = v6;
            v8[2].m128i_i32[0] = 544502639;
            *v8 = si128;
            v12 = _mm_load_si128((const __m128i *)&xmmword_4289520);
            v8[2].m128i_i16[2] = 26223;
            v8[1] = v12;
            *(_QWORD *)(v6 + 24) += 39LL;
          }
          sub_16E7A90(v11, v7);
          v13 = *(_QWORD *)(v6 + 24);
LABEL_11:
          if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v13) <= 0x11 )
            goto LABEL_30;
          goto LABEL_12;
        }
LABEL_28:
        if ( v9 <= 0x17 )
        {
          sub_16E7EE0(v6, "Cost Model: Unknown cost", 24);
          v13 = *(_QWORD *)(v6 + 24);
          goto LABEL_11;
        }
        v29 = _mm_load_si128((const __m128i *)&xmmword_4289530);
        v8[1].m128i_i64[0] = 0x74736F63206E776FLL;
        *v8 = v29;
        v13 = *(_QWORD *)(v6 + 24) + 24LL;
        v30 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v6 + 24) = v13;
        if ( (unsigned __int64)(v30 - v13) <= 0x11 )
        {
LABEL_30:
          v15 = sub_16E7EE0(v6, " for instruction: ", 18);
          goto LABEL_13;
        }
LABEL_12:
        v14 = _mm_load_si128((const __m128i *)&xmmword_4289540);
        v15 = v6;
        *(_WORD *)(v13 + 16) = 8250;
        *(__m128i *)v13 = v14;
        *(_QWORD *)(v6 + 24) += 18LL;
LABEL_13:
        sub_155C2B0(v17, v15, 0);
        v16 = *(_BYTE **)(v15 + 24);
        if ( *(_BYTE **)(v15 + 16) == v16 )
        {
          sub_16E7EE0(v15, "\n", 1);
          v5 = *(_QWORD *)(v5 + 8);
          if ( v37 == v5 )
          {
LABEL_33:
            a2 = v6;
            goto LABEL_34;
          }
        }
        else
        {
          *v16 = 10;
          ++*(_QWORD *)(v15 + 24);
          v5 = *(_QWORD *)(v5 + 8);
          if ( v37 == v5 )
            goto LABEL_33;
        }
      }
      v7 = sub_14A4F70(v18, v17);
LABEL_7:
      v8 = *(__m128i **)(v6 + 24);
      v9 = *(_QWORD *)(v6 + 16) - (_QWORD)v8;
      if ( v7 != -1 )
        goto LABEL_8;
      goto LABEL_28;
    }
  }
  return result;
}
