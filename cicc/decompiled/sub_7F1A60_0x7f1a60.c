// Function: sub_7F1A60
// Address: 0x7f1a60
//
void __fastcall sub_7F1A60(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int *a8)
{
  __int64 *v8; // r12
  __int64 v10; // rdi
  __m128i *v11; // rdx
  __m128i *v12; // rax
  __int64 v13; // r15
  int v14; // r13d
  __m128i *v15; // r15
  void *v16; // rax
  _QWORD *v17; // rax
  const __m128i *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  void *v22; // rax
  const __m128i *v23; // rsi
  void *v24; // rax
  __m128i *v25; // rax
  __int64 v26; // rax
  const __m128i *v27; // rax
  const __m128i *v28; // rsi
  int v29; // [rsp+Ch] [rbp-44h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __m128i *v32; // [rsp+18h] [rbp-38h]
  __m128i *v33; // [rsp+18h] [rbp-38h]
  void *v34; // [rsp+18h] [rbp-38h]

  v8 = (__int64 *)a4;
  v10 = a2;
  v29 = a6;
  if ( !(_DWORD)a6 || !a1 )
    goto LABEL_8;
  v11 = 0;
  while ( 1 )
  {
    v12 = (__m128i *)a1[1].m128i_i64[0];
    a1[1].m128i_i64[0] = (__int64)v11;
    v11 = a1;
    if ( !v12 )
      break;
    a1 = v12;
  }
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
LABEL_8:
      ;
    }
    while ( *(_BYTE *)(v10 + 140) == 12 );
  }
  if ( !a4 )
    v8 = sub_7DF5F0(v10, a3);
  if ( (_DWORD)a5 )
  {
    if ( !a1 )
      return;
    v31 = 0;
    v13 = (__int64)a1;
    while ( 2 )
    {
      if ( v31 || *(_QWORD *)(v13 + 16) )
      {
        while ( 1 )
        {
          v14 = sub_731920(v13, 1, 0, a4, a5, a6);
          if ( v14 )
            break;
          if ( v31 == 1 )
          {
            v14 = 1;
            goto LABEL_24;
          }
          v13 = *(_QWORD *)(v13 + 16);
          v31 = 1;
          if ( !v13 )
            goto LABEL_24;
        }
        v13 = *(_QWORD *)(v13 + 16);
        if ( v13 )
          continue;
      }
      break;
    }
  }
  else if ( !a1 )
  {
    return;
  }
  v14 = 0;
LABEL_24:
  v15 = a1;
  do
  {
    sub_7EE560(v15, 0);
    if ( (v15[1].m128i_i8[9] & 1) != 0 )
    {
      v22 = sub_730FF0(v15);
      v23 = (const __m128i *)sub_73E1B0((__int64)v22, 0);
      sub_730620((__int64)v15, v23);
      if ( v8 )
      {
LABEL_31:
        if ( (v8[4] & 1) != 0 && (!a3 || (*(_BYTE *)(a3 + 198) & 0x20) == 0) )
        {
          if ( (v15[1].m128i_i8[9] & 1) == 0 && (unsigned int)sub_8D3A70(v15->m128i_i64[0]) )
          {
            v27 = (const __m128i *)sub_730FF0(v15);
            v28 = sub_7EC130(v27);
            sub_730620((__int64)v15, v28);
          }
          if ( (v8[4] & 0x3F800) != 0 )
          {
            v34 = sub_730FF0(v15);
            v26 = sub_7E4BA0((__int64)v8);
            sub_7E2300((__int64)v15, (__int64)v34, v26);
          }
        }
        if ( unk_4F06968 )
        {
          sub_7F8570(v15);
        }
        else if ( dword_4F077C4 == 2
               && !(unsigned int)sub_7E1F90(v15->m128i_i64[0])
               && (unsigned int)sub_7E6740(v15->m128i_i64[0]) )
        {
          v33 = sub_7E6760((const __m128i *)v15->m128i_i64[0], v8[1]);
          v24 = sub_730FF0(v15);
          sub_7E2300((__int64)v15, (__int64)v24, (__int64)v33);
        }
        v8 = (__int64 *)*v8;
        goto LABEL_40;
      }
    }
    else if ( v8 )
    {
      goto LABEL_31;
    }
    if ( (unsigned int)sub_7E1F40(v15->m128i_i64[0]) )
      sub_7F8400(v15);
LABEL_40:
    if ( v14 && !sub_7175E0((__int64)v15, 0) && !(unsigned int)sub_731920((__int64)v15, 1, 0, v19, v20, v21)
      || a7 && (sub_7E6EE0(a7, (__int64)v15) || sub_7E6EE0((__int64)v15, a7)) )
    {
      v32 = sub_7E7CA0(v15->m128i_i64[0]);
      v16 = sub_730FF0(v15);
      v17 = (_QWORD *)sub_7E2BE0((__int64)v32, (__int64)v16);
      sub_7E69E0(v17, a8);
      v18 = (const __m128i *)sub_73E830((__int64)v32);
      sub_730620((__int64)v15, v18);
    }
    v15 = (__m128i *)v15[1].m128i_i64[0];
  }
  while ( v15 );
  if ( v29 )
  {
    while ( 1 )
    {
      v25 = (__m128i *)a1[1].m128i_i64[0];
      a1[1].m128i_i64[0] = (__int64)v15;
      v15 = a1;
      if ( !v25 )
        break;
      a1 = v25;
    }
  }
}
