// Function: sub_7FBC20
// Address: 0x7fbc20
//
_BYTE *__fastcall sub_7FBC20(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned int a4,
        __m128i *a5,
        unsigned int a6,
        __int64 a7)
{
  __int64 v8; // rbx
  unsigned __int8 v9; // al
  __m128i *v10; // rax
  int v11; // edx
  __m128i *v12; // r13
  __m128i *v13; // rax
  __int64 *v14; // r13
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // rax
  _BYTE *result; // rax
  __m128i *v22; // rax
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // r15d
  __int64 v30; // rcx
  __int64 v31; // rax
  char i; // dl
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __m128i *v38; // rdi
  __int64 v39; // r8
  __int64 v40; // r9
  __m128i *v41; // rsi
  __int64 v42; // r9
  const __m128i *v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  const __m128i *v46; // r13
  __int64 v47; // rsi
  _BYTE *v48; // rax
  _BYTE *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // eax
  unsigned int v55; // [rsp+0h] [rbp-60h]
  __m128i *v56; // [rsp+0h] [rbp-60h]
  __m128i *v57; // [rsp+0h] [rbp-60h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __m128i *v62; // [rsp+20h] [rbp-40h] BYREF
  __m128i *v63[7]; // [rsp+28h] [rbp-38h] BYREF

  v8 = a2;
  v61 = *a3;
  if ( !a1 )
  {
LABEL_7:
    if ( (*(_QWORD *)(v8 + 168) & 0xFF0000400008LL) == 0x20000000000LL )
    {
      v22 = (__m128i *)sub_73A720((const __m128i *)v8, a2);
      v22[1].m128i_i8[9] |= 1u;
      v12 = v22;
      a2 = dword_4D047E0;
      if ( dword_4D047E0 )
      {
        v57 = (__m128i *)sub_7259C0(8);
        sub_73C230(*(const __m128i **)(v8 + 128), v57);
        v43 = (const __m128i *)sub_8D4050(*(_QWORD *)(v8 + 128));
        v57[10].m128i_i64[0] = (__int64)sub_73C570(v43, 1);
        a2 = (__int64)v57;
        v12 = (__m128i *)sub_73DC90(v12, (__int64)v57);
      }
    }
    else
    {
      v10 = (__m128i *)sub_7EBB70(v8);
      v11 = 0;
      v12 = v10;
      if ( (v10[1].m128i_i8[9] & 1) == 0 )
      {
        if ( *(_BYTE *)(a7 + 17) )
          goto LABEL_10;
LABEL_24:
        v55 = v11;
        v23 = sub_8D3A70(v61);
        v27 = v55;
        if ( !v23 )
        {
LABEL_28:
          v29 = (_DWORD)v27 == 0 ? 73 : 86;
          goto LABEL_29;
        }
        goto LABEL_25;
      }
    }
LABEL_20:
    for ( result = (_BYTE *)v12->m128i_i64[0]; result[140] == 12; result = (_BYTE *)*((_QWORD *)result + 20) )
      ;
    if ( !*((_QWORD *)result + 22) )
      return result;
    v11 = 1;
    if ( *(_BYTE *)(a7 + 17) )
    {
LABEL_10:
      if ( !a1 )
        goto LABEL_45;
      goto LABEL_11;
    }
    goto LABEL_24;
  }
  v9 = *(_BYTE *)(a1 + 48);
  if ( v9 == 3 )
  {
    v12 = *(__m128i **)(a1 + 56);
    if ( (unsigned int)sub_8D3410(v61) )
      goto LABEL_20;
  }
  else
  {
    if ( v9 > 3u )
    {
      if ( (unsigned __int8)(v9 - 8) <= 1u )
        goto LABEL_6;
      goto LABEL_17;
    }
    if ( v9 != 1 )
    {
      if ( v9 == 2 )
      {
LABEL_6:
        v8 = *(_QWORD *)(a1 + 56);
        goto LABEL_7;
      }
LABEL_17:
      sub_721090();
    }
    v41 = (__m128i *)sub_724DC0();
    v62 = v41;
    sub_72BB40(v61, v41);
    v12 = (__m128i *)sub_73A720(v41, (__int64)v41);
    v56 = (__m128i *)v12[3].m128i_i64[1];
    if ( (unsigned int)sub_7E1F40(v61) )
    {
      if ( v56[10].m128i_i8[13] == 7 )
      {
        v56[-1].m128i_i8[8] &= ~8u;
        sub_7EAFC0(v56);
      }
      else
      {
        sub_72BAF0((__int64)v56, -1, 5u);
      }
      v42 = (__int64)v56;
    }
    else
    {
      v54 = sub_8D2B50(v61);
      v42 = (__int64)v56;
      if ( v54 )
      {
        v56[-1].m128i_i8[8] &= ~8u;
        sub_7EB190((__int64)v56, v41);
        v42 = (__int64)v56;
      }
    }
    a2 = (__int64)v63;
    if ( (unsigned int)sub_7EBAB0(v42, v63) )
    {
      a2 = 3;
      sub_7264E0((__int64)v12, 3);
      v12[3].m128i_i64[1] = (__int64)v63[0]->m128i_i64;
    }
    sub_724E30((__int64)&v62);
  }
  if ( *(_BYTE *)(a7 + 17) )
  {
LABEL_11:
    if ( *(_BYTE *)(a1 + 48) != 2 )
    {
      v13 = sub_7E7ED0(v12);
LABEL_13:
      v14 = 0;
      v15 = sub_731250((__int64)v13);
      v16 = sub_73E1B0((__int64)v15, a2);
      v17 = *(_QWORD *)(a7 + 40);
      v18 = (__int64)v16;
      if ( !v17 )
      {
        v44 = sub_7F9160(a7);
        v17 = *(_QWORD *)(a7 + 40);
        v14 = v44;
      }
      v19 = (__int64 *)sub_73E1B0((__int64)a3, a2);
      v20 = sub_7F9140(a7);
      return (_BYTE *)sub_7FB7C0(v20, a4, v19, v14, v17, v18, a5);
    }
LABEL_45:
    a2 = 0;
    v13 = sub_7EB890(v8, 0);
    goto LABEL_13;
  }
  v29 = 73;
  if ( !(unsigned int)sub_8D3A70(v61) )
  {
LABEL_29:
    if ( dword_4F077C4 != 2 || dword_4F06968 || (unsigned int)sub_7E1F90(v61) || !(unsigned int)sub_7E6740(v61) )
    {
      v30 = a6;
      if ( !a6
        || !(unsigned int)sub_8D32B0(v12->m128i_i64[0])
        || (v45 = sub_8D46C0(v12->m128i_i64[0]), !(unsigned int)sub_8D23B0(v45)) )
      {
LABEL_31:
        v31 = v12->m128i_i64[0];
        for ( i = *(_BYTE *)(v12->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v31 + 140) )
          v31 = *(_QWORD *)(v31 + 160);
        if ( (unsigned __int8)(i - 9) <= 2u )
        {
          v33 = *(_QWORD *)(v31 + 168);
          if ( (*(_BYTE *)(v33 + 109) & 0x10) != 0 )
          {
            v34 = *(_QWORD *)(v33 + 208);
            v35 = v61;
            if ( *(_BYTE *)(v61 + 140) == 12 )
            {
              do
                v35 = *(_QWORD *)(v35 + 160);
              while ( *(_BYTE *)(v35 + 140) == 12 );
            }
            else
            {
              v35 = v61;
            }
            if ( v34 == v35 )
            {
              v46 = sub_7EC2A0(v12, a2);
              v47 = sub_72D2E0((_QWORD *)v61);
              v48 = sub_73E110((__int64)v46, v47);
              v49 = sub_73DCD0(v48);
              v12 = (__m128i *)sub_731370((__int64)v49, v47, v50, v51, v52, v53);
            }
          }
        }
        v38 = (__m128i *)sub_698020(a3, v29, (__int64)v12, v30, v25, v26);
        if ( !a4 && v38[3].m128i_i8[9] == 10 )
          sub_7FA680(v38, v29, v36, v37, v39, v40);
        result = sub_7E69E0(v38, a5->m128i_i32);
        if ( result )
        {
          *(_QWORD *)result = *(_QWORD *)dword_4D03F38;
          *((_QWORD *)result + 1) = *(_QWORD *)dword_4D03F38;
        }
        return result;
      }
      a2 = v61;
    }
    else
    {
      a2 = (__int64)sub_7E6760((const __m128i *)v12->m128i_i64[0], v61);
    }
    if ( a2 )
    {
      if ( (v12[1].m128i_i8[9] & 1) != 0 )
        v12 = (__m128i *)sub_73DC50((__int64)v12, a2);
      else
        v12 = (__m128i *)sub_73E110((__int64)v12, a2);
    }
    goto LABEL_31;
  }
  v27 = 0;
LABEL_25:
  v28 = v61;
  if ( *(_BYTE *)(v61 + 140) == 12 )
  {
    do
      v28 = *(_QWORD *)(v28 + 160);
    while ( *(_BYTE *)(v28 + 140) == 12 );
  }
  else
  {
    v28 = v61;
  }
  if ( (*(_BYTE *)(v28 + 179) & 1) == 0 )
    goto LABEL_28;
  result = (_BYTE *)sub_731770((__int64)v12, 0, v27, v24, v25, v26);
  if ( (_DWORD)result )
    return sub_7E69E0(v12, a5->m128i_i32);
  return result;
}
