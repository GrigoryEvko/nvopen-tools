// Function: sub_857CE0
// Address: 0x857ce0
//
_QWORD *sub_857CE0()
{
  _QWORD *result; // rax
  __m128i *v1; // r15
  __int64 *v2; // rbx
  __m128i *v3; // r14
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  bool v11; // cf
  bool v12; // zf
  __int64 v13; // rdx
  __int64 v14; // rcx
  char *v15; // rdi
  unsigned int *v16; // r10
  unsigned int *v17; // rsi
  char v18; // al
  bool v19; // cf
  bool v20; // zf
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rdx
  _BYTE *v25; // rax
  bool v26; // cf
  bool v27; // zf
  char v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  bool v34; // cf
  bool v35; // zf
  bool v36; // cf
  bool v37; // zf
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  char *v46; // [rsp+0h] [rbp-40h]
  char v47; // [rsp+Fh] [rbp-31h]

  result = &qword_4D03E88;
  v1 = (__m128i *)qword_4D03E88;
  if ( qword_4D03E88 )
  {
    v2 = 0;
    do
    {
      while ( 1 )
      {
        v3 = v1;
        v1 = (__m128i *)v1->m128i_i64[0];
        result = (_QWORD *)v3->m128i_i64[1];
        if ( *((_BYTE *)result + 8) == 26 )
          break;
        v2 = (__int64 *)v3;
        if ( !v1 )
          return result;
      }
      sub_7C9660((__int64)v3);
      if ( word_4F06418[0] != 1 )
      {
LABEL_6:
        sub_684AC0(byte_4F07472[0], 0x40Fu);
        sub_7C96B0(1u, (unsigned int *)0x40F, v6, v7, v8, v9);
        goto LABEL_7;
      }
      if ( dword_4F077C4 == 2 )
      {
        v11 = unk_4F07778 < 0x3118Eu;
        v12 = unk_4F07778 == 201102;
        if ( unk_4F07778 <= 201102 )
        {
          v11 = 0;
          v12 = dword_4F07774 == 0;
          if ( !dword_4F07774 )
            goto LABEL_6;
        }
      }
      else
      {
        v11 = unk_4F07778 < 0x30CDCu;
        v12 = unk_4F07778 == 199900;
        if ( unk_4F07778 <= 199900 )
          goto LABEL_6;
      }
      v13 = (__int64)&qword_4D04A00;
      v14 = 12;
      v15 = "FP_CONTRACT";
      v16 = *(unsigned int **)(qword_4D04A00 + 8);
      v17 = v16;
      do
      {
        if ( !v14 )
          break;
        v11 = *(_BYTE *)v17 < (unsigned __int8)*v15;
        v12 = *(_BYTE *)v17 == (unsigned __int8)*v15;
        v17 = (unsigned int *)((char *)v17 + 1);
        ++v15;
        --v14;
      }
      while ( v12 );
      v18 = (!v11 && !v12) - v11;
      v19 = 0;
      v20 = v18 == 0;
      if ( v18 )
      {
        v14 = 12;
        v17 = *(unsigned int **)(qword_4D04A00 + 8);
        v15 = "FENV_ACCESS";
        do
        {
          if ( !v14 )
            break;
          v19 = *(_BYTE *)v17 < (unsigned __int8)*v15;
          v20 = *(_BYTE *)v17 == (unsigned __int8)*v15;
          v17 = (unsigned int *)((char *)v17 + 1);
          ++v15;
          --v14;
        }
        while ( v20 );
        if ( (!v19 && !v20) == v19 )
        {
          v47 = 2;
          v46 = (char *)&unk_4F06C59;
        }
        else
        {
          v13 = (unsigned int)qword_4F077B4;
          v34 = 0;
          v35 = (_DWORD)qword_4F077B4 == 0;
          if ( (_DWORD)qword_4F077B4 )
            goto LABEL_6;
          v14 = 17;
          v15 = "CX_LIMITED_RANGE";
          v17 = v16;
          do
          {
            if ( !v14 )
              break;
            v34 = *(_BYTE *)v17 < (unsigned __int8)*v15;
            v35 = *(_BYTE *)v17 == (unsigned __int8)*v15;
            v17 = (unsigned int *)((char *)v17 + 1);
            ++v15;
            --v14;
          }
          while ( v35 );
          if ( (!v34 && !v35) != v34 )
            goto LABEL_6;
          v47 = 3;
          v46 = (char *)&unk_4F06C58;
        }
      }
      else
      {
        v47 = 1;
        v46 = (char *)&unk_4F06C5A;
      }
      sub_7B8B50((unsigned __int64)v15, v17, v13, v14, v4, v5);
      v24 = &qword_4D04A00;
      if ( word_4F06418[0] != 1 )
        goto LABEL_47;
      v25 = *(_BYTE **)(qword_4D04A00 + 8);
      v26 = *v25 < 0x4Fu;
      v27 = *v25 == 79;
      if ( *v25 == 79 && (v26 = v25[1] < 0x4Eu, v27 = v25[1] == 78) && (v26 = 0, v27 = v25[2] == 0, !v25[2]) )
      {
        if ( v47 == 2 )
        {
          v28 = 2;
          if ( (_DWORD)qword_4F077B4 && qword_4F077A0 <= 0x1D4BFu )
          {
            sub_684AC0(7u, 0xB5Fu);
            sub_7C96B0(1u, (unsigned int *)0xB5F, v42, v43, v44, v45);
LABEL_7:
            v10 = v3->m128i_i64[0];
            if ( !v2 )
              goto LABEL_28;
            goto LABEL_8;
          }
        }
        else
        {
          v28 = 2;
        }
      }
      else
      {
        v21 = 4;
        v17 = *(unsigned int **)(qword_4D04A00 + 8);
        v15 = "OFF";
        do
        {
          if ( !v21 )
            break;
          v26 = *(_BYTE *)v17 < (unsigned __int8)*v15;
          v27 = *(_BYTE *)v17 == (unsigned __int8)*v15;
          v17 = (unsigned int *)((char *)v17 + 1);
          ++v15;
          --v21;
        }
        while ( v27 );
        LOBYTE(v24) = (!v26 && !v27) - v26;
        v36 = 0;
        v37 = (_BYTE)v24 == 0;
        if ( (_BYTE)v24 )
        {
          v17 = *(unsigned int **)(qword_4D04A00 + 8);
          v21 = 8;
          v15 = "DEFAULT";
          do
          {
            if ( !v21 )
              break;
            v36 = *(_BYTE *)v17 < (unsigned __int8)*v15;
            v37 = *(_BYTE *)v17 == (unsigned __int8)*v15;
            v17 = (unsigned int *)((char *)v17 + 1);
            ++v15;
            --v21;
          }
          while ( v37 );
          if ( (!v36 && !v37) != v36 )
          {
LABEL_47:
            sub_684AC0(byte_4F07472[0], 0x410u);
            sub_7C96B0(1u, (unsigned int *)0x410, v38, v39, v40, v41);
            goto LABEL_7;
          }
          v28 = 3;
        }
        else
        {
          v28 = 1;
        }
      }
      sub_7B8B50((unsigned __int64)v15, v17, (__int64)v24, v21, v22, v23);
      sub_7C96B0(0, v17, v29, v30, v31, v32);
      sub_8543B0(v3, 0, 0);
      v33 = v3[5].m128i_i64[1];
      if ( v33 )
      {
        *(_BYTE *)(v33 + 56) = v47;
        *(_BYTE *)(v3[5].m128i_i64[1] + 57) = v28;
      }
      *v46 = v28;
      v10 = v3->m128i_i64[0];
      if ( !v2 )
      {
LABEL_28:
        qword_4D03E88 = v10;
        goto LABEL_9;
      }
LABEL_8:
      *v2 = v10;
LABEL_9:
      result = sub_853F90(v3);
    }
    while ( v1 );
  }
  return result;
}
