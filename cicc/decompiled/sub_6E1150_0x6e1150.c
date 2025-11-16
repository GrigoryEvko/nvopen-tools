// Function: sub_6E1150
// Address: 0x6e1150
//
__int64 __fastcall sub_6E1150(_QWORD *a1, unsigned int a2, __int64 *a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r13
  __int64 v6; // rax
  __m128i *v7; // rbx
  bool v8; // zf
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rbx
  int v12; // eax
  __int64 result; // rax
  int v14; // eax
  _QWORD *v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // r14
  _QWORD *v18; // rdx
  _QWORD *v19; // r13
  int v20; // r14d
  __int64 m128i_i64; // r8
  const __m128i *v22; // r12
  const __m128i *v23; // r15
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  __int64 v27; // r9
  __int64 v28; // rbx
  int v29; // eax
  const __m128i *v30; // r15
  unsigned int i; // r12d
  __m128i *v32; // rbx
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // rdi
  __int64 v36; // rdi
  __int64 v37; // r9
  int v38; // eax
  __int64 v39; // rdi
  __int64 v40; // [rsp+0h] [rbp-A0h]
  _QWORD *v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  const __m128i *v44; // [rsp+18h] [rbp-88h]
  __int64 v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  const __m128i *v47; // [rsp+20h] [rbp-80h]
  unsigned int v48; // [rsp+2Ch] [rbp-74h]
  __int64 v49; // [rsp+30h] [rbp-70h]
  __int64 v50; // [rsp+38h] [rbp-68h]
  __m128i v51; // [rsp+60h] [rbp-40h] BYREF

  v5 = a1;
  v6 = a3[1];
  v7 = (__m128i *)(*a1 + 48LL * a2);
  v51.m128i_i64[0] = a4;
  v8 = v7->m128i_i64[0] == 0;
  v9 = *a3;
  v51.m128i_i64[1] = a5;
  v49 = v6;
  v10 = a3[3];
  v50 = a3[2];
  if ( v8 && !v7[1].m128i_i64[0] )
  {
    v35 = v7->m128i_i64[1];
    if ( v35 )
      sub_7386E0(v35, 0, 7, a4, a5);
  }
  v7->m128i_i64[0] = v9;
  v7[1].m128i_i64[1] = v10;
  v7->m128i_i64[1] = v49;
  v7[1].m128i_i64[0] = v50;
  if ( v9 || v50 )
  {
    v7[2] = _mm_loadu_si128(&v51);
    v11 = *((unsigned int *)v5 + 2);
    v12 = *((_DWORD *)v5 + 3) + 1;
    *((_DWORD *)v5 + 3) = v12;
    result = (unsigned int)(2 * v12);
    if ( (unsigned int)result <= (unsigned int)v11 )
      return result;
  }
  else
  {
    if ( v49 && !(unsigned int)sub_7386E0(v49, 0, 7, a4, a5) || v10 )
      v7[2] = _mm_loadu_si128(&v51);
    v11 = *((unsigned int *)v5 + 2);
    v14 = *((_DWORD *)v5 + 3) + 1;
    *((_DWORD *)v5 + 3) = v14;
    if ( (unsigned int)v11 >= 2 * v14 )
    {
LABEL_31:
      result = v49;
      if ( v49 )
        return sub_7386E0(v49, 0, 7, a4, a5);
      return result;
    }
  }
  v48 = v11 + 1;
  v15 = (_QWORD *)sub_823970(48LL * (unsigned int)(2 * v11 + 2));
  v16 = (unsigned int)(2 * v11 + 1);
  v17 = v15;
  if ( 2 * (_DWORD)v11 != -2 )
  {
    v18 = &v15[6 * (unsigned int)v16 + 6];
    do
    {
      if ( v15 )
      {
        *v15 = 0;
        v15[1] = 0;
        v15[2] = 0;
        v15[3] = 0;
      }
      v15 += 6;
    }
    while ( v18 != v15 );
  }
  v47 = (const __m128i *)*v5;
  if ( (_DWORD)v11 != -1 )
  {
    v42 = v9;
    v41 = v5;
    v19 = v17;
    v20 = 2 * v11 + 1;
    m128i_i64 = (__int64)v47[3 * v11 + 3].m128i_i64;
    v22 = v47;
    v23 = (const __m128i *)m128i_i64;
    while ( 1 )
    {
      v24 = v22->m128i_i64[0];
      v25 = v22->m128i_i64[1];
      v26 = v22[1].m128i_u64[0];
      v27 = v22[1].m128i_i64[1];
      if ( v22->m128i_i64[0] || v26 )
        goto LABEL_17;
      if ( v25 )
      {
        v45 = v22[1].m128i_i64[1];
        if ( !(unsigned int)sub_7386E0(v25, 0, 7, v16, m128i_i64) )
          break;
        v27 = v45;
      }
      if ( v27 )
        break;
LABEL_27:
      v22 += 3;
      if ( v22 == v23 )
      {
        LODWORD(v16) = v20;
        v9 = v42;
        v17 = v19;
        v5 = v41;
        goto LABEL_29;
      }
    }
    v25 = v22->m128i_i64[1];
    v26 = v22[1].m128i_u64[0];
    v24 = v22->m128i_i64[0];
    v27 = v22[1].m128i_i64[1];
LABEL_17:
    v43 = v27;
    v28 = 31 * (31 * (31 * ((v24 >> 3) + 527) + (v26 >> 3)) + (unsigned int)sub_72A8B0(v25));
    v29 = sub_72E120(v43);
    v44 = v23;
    v30 = v22;
    for ( i = v20 & (v29 + v28); ; i = v20 & (i + 1) )
    {
      v32 = (__m128i *)&v19[6 * i];
      if ( !v32->m128i_i64[0] && !v32[1].m128i_i64[0] )
      {
        v33 = v32->m128i_i64[1];
        m128i_i64 = v32[1].m128i_i64[1];
        if ( !v33 || (v40 = v32[1].m128i_i64[1], v34 = sub_7386E0(v33, 0, 7, v16, m128i_i64), m128i_i64 = v40, v34) )
        {
          if ( !m128i_i64 )
            break;
        }
      }
    }
    v22 = v30;
    v23 = v44;
    if ( !v32->m128i_i64[0] && !v32[1].m128i_i64[0] )
    {
      v39 = v32->m128i_i64[1];
      if ( v39 )
        sub_7386E0(v39, 0, 7, v16, 0);
    }
    *v32 = _mm_loadu_si128(v22);
    v8 = v32->m128i_i64[0] == 0;
    v32[1] = _mm_loadu_si128(v22 + 1);
    if ( !v8
      || v32[1].m128i_i64[0]
      || (v36 = v32->m128i_i64[1], v37 = v32[1].m128i_i64[1], v36)
      && (v46 = v32[1].m128i_i64[1], v38 = sub_7386E0(v36, 0, 7, v16, m128i_i64), v37 = v46, !v38)
      || v37 )
    {
      v32[2] = _mm_loadu_si128(v22 + 2);
    }
    goto LABEL_27;
  }
LABEL_29:
  *v5 = v17;
  *((_DWORD *)v5 + 2) = v16;
  result = sub_823A00(v47, 48LL * v48);
  if ( !v9 && !v50 )
    goto LABEL_31;
  return result;
}
