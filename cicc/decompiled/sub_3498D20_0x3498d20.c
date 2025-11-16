// Function: sub_3498D20
// Address: 0x3498d20
//
__int64 __fastcall sub_3498D20(
        __int64 a1,
        unsigned __int64 *a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  __int64 (*v7)(); // rax
  unsigned __int16 v8; // dx
  unsigned __int16 j; // r12
  unsigned __int64 v10; // r13
  char v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r10
  unsigned int v20; // r15d
  __int64 v21; // rsi
  __int64 v22; // rdx
  char v23; // al
  unsigned __int64 v24; // rax
  int v25; // eax
  __int64 (*v26)(); // rdx
  unsigned __int16 i; // ax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 (*v31)(); // rax
  __int16 k; // si
  __int16 v33; // si
  __int64 (*v34)(); // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // r8
  __int64 (*v41)(); // rax
  __m128i *v42; // rsi
  char v43; // al
  __int64 (*v44)(); // rax
  __int64 v45; // rdx
  char v46; // al
  __int64 v47; // rdx
  unsigned int v49; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v50; // [rsp+8h] [rbp-D8h]
  int v54; // [rsp+2Ch] [rbp-B4h]
  __m128i v55; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+50h] [rbp-90h] BYREF
  char v58; // [rsp+58h] [rbp-88h]
  __int64 v59; // [rsp+60h] [rbp-80h]
  __int64 v60; // [rsp+68h] [rbp-78h]
  __int64 v61; // [rsp+70h] [rbp-70h]
  __int64 v62; // [rsp+78h] [rbp-68h]
  __int64 v63; // [rsp+80h] [rbp-60h]
  __int64 v64; // [rsp+88h] [rbp-58h]
  __int64 v65; // [rsp+90h] [rbp-50h]
  __int64 v66; // [rsp+98h] [rbp-48h]
  __int64 v67; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-38h]

  if ( a3 != -1 && !*(_BYTE *)(a4 + 11) && !*(_BYTE *)(a4 + 8) && *(_BYTE *)(a4 + 14) < *(_BYTE *)(a4 + 9) )
    return 0;
  v7 = *(__int64 (**)())(*(_QWORD *)a1 + 832LL);
  if ( v7 == sub_2FE32C0
    || (v55.m128i_i32[0] = ((__int64 (__fastcall *)(__int64, __int64, __int64))v7)(a1, a4, a7),
        v55.m128i_i64[1] = v47,
        v55.m128i_i16[0] == 1) )
  {
    v55.m128i_i64[1] = 0;
    v55.m128i_i16[0] = 9;
    if ( *(_BYTE *)(a4 + 8) )
    {
      v8 = 9;
    }
    else
    {
      for ( i = 9; ; i = --v55.m128i_i16[0] )
      {
        if ( i )
        {
          if ( i == 1 || (unsigned __int16)(i - 504) <= 7u )
            goto LABEL_96;
          v30 = 16LL * (i - 1);
          v29 = *(_QWORD *)&byte_444C4A0[v30];
          LOBYTE(v30) = byte_444C4A0[v30 + 8];
        }
        else
        {
          v29 = sub_3007260((__int64)&v55);
          v59 = v29;
          v60 = v30;
        }
        LOBYTE(v68) = v30;
        v67 = v29;
        if ( (unsigned __int64)sub_CA1930(&v67) >> 3 <= 1LL << *(_BYTE *)(a4 + 9) )
          break;
        v31 = *(__int64 (**)())(*(_QWORD *)a1 + 808LL);
        if ( v31 != sub_2D56600 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD))v31)(
                 a1,
                 v55.m128i_u32[0],
                 v55.m128i_i64[1],
                 a5,
                 *(unsigned __int8 *)(a4 + 9),
                 0,
                 0) )
          {
            break;
          }
        }
        v55.m128i_i64[1] = 0;
      }
      v8 = v55.m128i_i16[0];
    }
    for ( j = 9; !j || !*(_QWORD *)(a1 + 8LL * j + 112); --j )
      ;
    if ( j != v8 )
    {
      if ( j == 1 || (unsigned __int16)(j - 504) <= 7u )
        goto LABEL_96;
      v10 = *(_QWORD *)&byte_444C4A0[16 * j - 16];
      v11 = byte_444C4A0[16 * j - 8];
      if ( v8 )
      {
        if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
          goto LABEL_96;
        v14 = 16LL * (v8 - 1) + 71615648;
        v15 = *(_QWORD *)&byte_444C4A0[16 * v8 - 16];
        LOBYTE(v14) = *(_BYTE *)(v14 + 8);
      }
      else
      {
        v12 = sub_3007260((__int64)&v55);
        v14 = v13;
        v61 = v12;
        v15 = v12;
        v62 = v14;
      }
      if ( ((_BYTE)v14 || !v11) && v10 < v15 )
      {
        v55.m128i_i16[0] = j;
        v55.m128i_i64[1] = 0;
      }
    }
  }
  v54 = 0;
  v16 = *(_QWORD *)a4;
  if ( *(_QWORD *)a4 )
  {
    while ( 1 )
    {
      if ( v55.m128i_i16[0] )
      {
        if ( v55.m128i_i16[0] == 1 || (unsigned __int16)(v55.m128i_i16[0] - 504) <= 7u )
          goto LABEL_96;
        v18 = 16LL * (v55.m128i_u16[0] - 1);
        v17 = *(_QWORD *)&byte_444C4A0[v18];
        LOBYTE(v18) = byte_444C4A0[v18 + 8];
      }
      else
      {
        v17 = sub_3007260((__int64)&v55);
        v63 = v17;
        v64 = v18;
      }
      v57 = v17;
      v58 = v18;
      v19 = (unsigned int)((unsigned __int64)sub_CA1930(&v57) >> 3);
      if ( v16 < v19 )
        break;
LABEL_57:
      if ( a3 < ++v54 )
        return 0;
      v42 = (__m128i *)a2[1];
      if ( v42 == (__m128i *)a2[2] )
      {
        v50 = v19;
        sub_3498BA0(a2, v42, &v55);
        v19 = v50;
      }
      else
      {
        if ( v42 )
        {
          *v42 = _mm_loadu_si128(&v55);
          v42 = (__m128i *)a2[1];
        }
        a2[1] = (unsigned __int64)&v42[1];
      }
      v16 -= v19;
      if ( !v16 )
        return 1;
    }
    while ( 1 )
    {
      LOWORD(v20) = v55.m128i_i16[0];
      v56 = _mm_loadu_si128(&v55);
      if ( v55.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v55.m128i_i16[0] - 10) > 0xDAu )
          goto LABEL_43;
        v21 = 16LL * (v55.m128i_u16[0] - 1);
        v22 = *(_QWORD *)&byte_444C4A0[v21];
        v23 = byte_444C4A0[v21 + 8];
      }
      else
      {
        if ( !sub_30070B0((__int64)&v55) && !(unsigned __int8)sub_3007030((__int64)&v55) )
        {
LABEL_43:
          k = v20;
          goto LABEL_44;
        }
        v65 = sub_3007260((__int64)&v55);
        v66 = v45;
        v22 = v65;
        v23 = v66;
      }
      v57 = v22;
      v58 = v23;
      v24 = sub_CA1930(&v57);
      v56.m128i_i64[1] = 0;
      v20 = 8 - (v24 < 0x41);
      v56.m128i_i16[0] = 8 - (v24 < 0x41);
      v25 = v20;
      if ( *(_QWORD *)(a1 + 8LL * (int)v20 + 112) && (*(_BYTE *)(a1 + 500LL * v20 + 6713) & 0xFB) == 0 )
      {
        v26 = *(__int64 (**)())(*(_QWORD *)a1 + 848LL);
        if ( v26 == sub_2FE32E0 )
          goto LABEL_28;
        v43 = ((__int64 (__fastcall *)(__int64, _QWORD))v26)(a1, v20);
        LOWORD(v20) = v56.m128i_i16[0];
        if ( v43 )
          goto LABEL_47;
      }
      if ( (_WORD)v20 != 8 || !*(_QWORD *)(a1 + 216) || (*(_BYTE *)(a1 + 13213) & 0xFB) != 0 )
        goto LABEL_43;
      v44 = *(__int64 (**)())(*(_QWORD *)a1 + 848LL);
      if ( v44 != sub_2FE32E0 && !((unsigned __int8 (__fastcall *)(__int64, __int64))v44)(a1, 13) )
      {
        for ( k = v56.m128i_i16[0]; ; k = v56.m128i_i16[0] )
        {
LABEL_44:
          v56.m128i_i64[1] = 0;
          v33 = k - 1;
          v56.m128i_i16[0] = v33;
          if ( v33 == 5 )
          {
            v25 = 5;
            goto LABEL_70;
          }
          v34 = *(__int64 (**)())(*(_QWORD *)a1 + 848LL);
          if ( v34 == sub_2FE32E0 )
          {
            LOWORD(v20) = v33;
            goto LABEL_47;
          }
          if ( ((unsigned __int8 (__fastcall *)(__int64))v34)(a1) )
            break;
        }
        LOWORD(v20) = v56.m128i_i16[0];
LABEL_47:
        if ( !(_WORD)v20 )
        {
          v35 = sub_3007260((__int64)&v56);
          v37 = v36;
          v67 = v35;
          v38 = v35;
          v68 = v37;
          goto LABEL_49;
        }
        v25 = (unsigned __int16)v20;
        if ( (_WORD)v20 == 1 )
LABEL_96:
          BUG();
LABEL_28:
        if ( (unsigned __int16)(v20 - 504) <= 7u )
          goto LABEL_96;
        goto LABEL_70;
      }
      v25 = 13;
      v56.m128i_i64[1] = 0;
      v56.m128i_i16[0] = 13;
LABEL_70:
      v37 = 16LL * (v25 - 1);
      v38 = *(_QWORD *)&byte_444C4A0[v37];
      LOBYTE(v37) = byte_444C4A0[v37 + 8];
LABEL_49:
      v57 = v38;
      v58 = v37;
      v39 = (unsigned __int64)sub_CA1930(&v57) >> 3;
      v19 = (unsigned int)v39;
      if ( !v54 || !*(_BYTE *)(a4 + 10) || (unsigned int)v39 >= v16 )
        goto LABEL_55;
      v40 = 0;
      v41 = *(__int64 (**)())(*(_QWORD *)a1 + 808LL);
      if ( !*(_BYTE *)(a4 + 8) )
        v40 = *(unsigned __int8 *)(a4 + 9);
      if ( v41 != sub_2D56600
        && (v49 = v19,
            v46 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, __int64 *))v41)(
                    a1,
                    v55.m128i_u32[0],
                    v55.m128i_i64[1],
                    a5,
                    v40,
                    0,
                    &v57),
            v19 = v49,
            v46)
        && (_DWORD)v57 )
      {
        v19 = (unsigned int)v16;
      }
      else
      {
LABEL_55:
        v55 = _mm_loadu_si128(&v56);
      }
      if ( v19 <= v16 )
        goto LABEL_57;
    }
  }
  return 1;
}
