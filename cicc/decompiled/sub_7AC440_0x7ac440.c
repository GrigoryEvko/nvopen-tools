// Function: sub_7AC440
// Address: 0x7ac440
//
int sub_7AC440()
{
  unsigned __int8 *v0; // r12
  int v1; // r15d
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rax
  char *v6; // r13
  unsigned __int64 v7; // rdi
  int v8; // eax
  _QWORD *v9; // rdi
  char *v10; // r15
  char *v11; // r12
  __int64 v12; // rax
  char v13; // cl
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  size_t v17; // rsi
  char *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  size_t v21; // rsi
  __int64 *v22; // rax
  __int64 v23; // r13
  __m128i **v24; // r12
  const char *v25; // r14
  __int64 v26; // rbx
  __m128i *v27; // rax
  __m128i *v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rdi
  char *p_key; // r13
  __int64 v32; // rax
  char v33; // cl
  __int64 v34; // rdx
  char *v36; // [rsp+0h] [rbp-A0h]
  int v37; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v38; // [rsp+8h] [rbp-98h]
  int v39; // [rsp+2Ch] [rbp-74h]
  int v40; // [rsp+30h] [rbp-70h] BYREF
  int key; // [rsp+34h] [rbp-6Ch] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-68h] BYREF
  char *s1[2]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+60h] [rbp-40h]

  v0 = (unsigned __int8 *)unk_4F06410;
  if ( qword_4F084A8 )
  {
    sub_823800(qword_4F084A0);
    sub_823800(qword_4F08498);
  }
  else
  {
    qword_4F084A8 = sub_881A70(0, 0x10000, 38, 39);
    qword_4F084A0 = sub_8237A0(256);
    qword_4F08498 = sub_8237A0(256);
  }
  LODWORD(v45) = 0;
  v1 = 0;
  v44.m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
  if ( (unsigned __int64)v0 <= unk_4F06408 )
  {
    while ( 1 )
    {
      v39 = sub_722680(v0, &v42, &v40, 0);
      v14 = v42;
      if ( v42 <= 0x7F )
      {
        v2 = qword_4F084A0;
        v3 = *(_QWORD *)(qword_4F084A0 + 16);
        if ( (unsigned __int64)(v3 + 1) > *(_QWORD *)(qword_4F084A0 + 8) )
        {
          sub_823810(qword_4F084A0);
          v2 = qword_4F084A0;
          v14 = v42;
          v3 = *(_QWORD *)(qword_4F084A0 + 16);
        }
        *(_BYTE *)(*(_QWORD *)(v2 + 32) + v3) = v14;
        ++*(_QWORD *)(v2 + 16);
        v4 = qword_4F08498;
        v5 = *(_QWORD *)(qword_4F08498 + 16);
        if ( (unsigned __int64)(v5 + 1) > *(_QWORD *)(qword_4F08498 + 8) )
        {
          sub_823810(qword_4F08498);
          v4 = qword_4F08498;
          v5 = *(_QWORD *)(qword_4F08498 + 16);
        }
        *(_BYTE *)(*(_QWORD *)(v4 + 32) + v5) = v42;
        ++*(_QWORD *)(v4 + 16);
        LODWORD(v45) = v45 + v42 + 32 * v45;
        goto LABEL_10;
      }
      key = v42;
      v36 = (char *)bsearch(&key, &unk_4AF8160, unk_4AF8158, 0x4Cu, (__compar_fn_t)sub_7AB720);
      v37 = sub_722A20(v42, &key);
      sub_8238B0(qword_4F084A0, &key, v37);
      if ( v42 - 8203 <= 2 )
        goto LABEL_23;
      v6 = v36 + 4;
      if ( v36 )
        break;
      if ( v37 > 0 )
      {
        v30 = (_QWORD *)qword_4F08498;
        p_key = (char *)&key;
        v32 = *(_QWORD *)(qword_4F08498 + 16);
        if ( (unsigned __int64)(v32 + 1) > *(_QWORD *)(qword_4F08498 + 8) )
          goto LABEL_41;
        while ( 1 )
        {
          do
          {
            v33 = *p_key++;
            *(_BYTE *)(v30[4] + v32) = v33;
            ++v30[2];
            LODWORD(v45) = *(p_key - 1) + 33 * v45;
            if ( (char *)&key + (unsigned int)(v37 - 1) + 1 == p_key )
              goto LABEL_10;
            v32 = v30[2];
          }
          while ( (unsigned __int64)(v32 + 1) <= v30[1] );
LABEL_41:
          sub_823810(v30);
          v30 = (_QWORD *)qword_4F08498;
          v32 = *(_QWORD *)(qword_4F08498 + 16);
        }
      }
LABEL_10:
      v0 += v39;
      if ( unk_4F06408 < (unsigned __int64)v0 )
        goto LABEL_24;
    }
    v38 = v0;
    do
    {
      v7 = *(int *)v6;
      if ( !(_DWORD)v7 )
        break;
      v8 = sub_722A20(v7, &key);
      if ( v8 > 0 )
      {
        v9 = (_QWORD *)qword_4F08498;
        v10 = (char *)&key;
        v11 = (char *)&key + (unsigned int)(v8 - 1) + 1;
        do
        {
          v12 = v9[2];
          if ( (unsigned __int64)(v12 + 1) > v9[1] )
          {
            sub_823810(v9);
            v9 = (_QWORD *)qword_4F08498;
            v12 = *(_QWORD *)(qword_4F08498 + 16);
          }
          v13 = *v10++;
          *(_BYTE *)(v9[4] + v12) = v13;
          ++v9[2];
          LODWORD(v45) = *(v10 - 1) + 33 * v45;
        }
        while ( v11 != v10 );
      }
      v6 += 4;
    }
    while ( v36 + 76 != v6 );
    v0 = v38;
LABEL_23:
    v1 = 1;
    goto LABEL_10;
  }
LABEL_24:
  v15 = qword_4F084A0;
  v16 = *(_QWORD *)(qword_4F084A0 + 16);
  if ( (unsigned __int64)(v16 + 1) > *(_QWORD *)(qword_4F084A0 + 8) )
  {
    sub_823810(qword_4F084A0);
    v15 = qword_4F084A0;
    v16 = *(_QWORD *)(qword_4F084A0 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v15 + 32) + v16) = 0;
  v17 = *(_QWORD *)(v15 + 16) + 1LL;
  *(_QWORD *)(v15 + 16) = v17;
  v18 = (char *)sub_7AC370(*(void **)(v15 + 32), v17);
  s1[1] = v18;
  if ( v1 )
  {
    v19 = qword_4F08498;
    v20 = *(_QWORD *)(qword_4F08498 + 16);
    if ( (unsigned __int64)(v20 + 1) > *(_QWORD *)(qword_4F08498 + 8) )
    {
      sub_823810(qword_4F08498);
      v19 = qword_4F08498;
      v20 = *(_QWORD *)(qword_4F08498 + 16);
    }
    *(_BYTE *)(*(_QWORD *)(v19 + 32) + v20) = 0;
    v21 = *(_QWORD *)(v19 + 16) + 1LL;
    *(_QWORD *)(v19 + 16) = v21;
    v18 = (char *)sub_7AC370(*(void **)(v19 + 32), v21);
  }
  v44.m128i_i64[0] = (__int64)v18;
  v22 = (__int64 *)sub_881B20(qword_4F084A8, s1, 1);
  v23 = *v22;
  v24 = (__m128i **)v22;
  if ( *v22 )
  {
    v25 = s1[1];
    v26 = *v22;
    while ( 1 )
    {
      LODWORD(v27) = strcmp(v25, *(const char **)(v26 + 8));
      v26 = *(_QWORD *)v26;
      if ( !v26 )
        break;
      if ( !(_DWORD)v27 )
        return (int)v27;
    }
    if ( (_DWORD)v27 )
    {
      v28 = (__m128i *)sub_823970(40);
      *v28 = _mm_loadu_si128((const __m128i *)s1);
      v28[1] = _mm_loadu_si128(&v44);
      v29 = v45;
      v28->m128i_i64[0] = v23;
      v28[2].m128i_i64[0] = v29;
      *v24 = v28;
      LODWORD(v27) = sub_6861C0(5u, 0xC9Bu, &dword_4F063F8, (_QWORD *)(v23 + 24), (__int64)v25);
    }
  }
  else
  {
    v27 = (__m128i *)sub_823970(40);
    *v27 = _mm_loadu_si128((const __m128i *)s1);
    v27[1] = _mm_loadu_si128(&v44);
    v34 = v45;
    v27->m128i_i64[0] = 0;
    v27[2].m128i_i64[0] = v34;
    *v24 = v27;
  }
  return (int)v27;
}
