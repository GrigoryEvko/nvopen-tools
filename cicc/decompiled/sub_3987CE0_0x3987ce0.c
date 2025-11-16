// Function: sub_3987CE0
// Address: 0x3987ce0
//
void __fastcall sub_3987CE0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // rbx
  __m128i *v4; // rbx
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  __int32 v8; // edx
  __int64 v9; // rax
  __m128i *v10; // rbx
  unsigned __int64 v11; // r13
  __int32 v12; // edx
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  __m128i v15; // xmm3
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // rbx
  __int32 v21; // edx
  unsigned __int64 v22; // r13
  __int32 v23; // edx
  __int64 v24; // rax
  __m128i v25; // xmm6
  __int64 v26; // [rsp+10h] [rbp-A0h]
  __m128i *v27; // [rsp+18h] [rbp-98h]
  char *v28; // [rsp+20h] [rbp-90h]
  unsigned __int64 v29; // [rsp+28h] [rbp-88h]
  __m128i *v30; // [rsp+30h] [rbp-80h]
  char v31[8]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-68h]
  char v33[8]; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v34; // [rsp+68h] [rbp-48h]

  v3 = (char *)a2 - (char *)a1;
  v27 = a2;
  v26 = a3;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return;
  if ( !a3 )
  {
    v28 = (char *)a2;
    goto LABEL_18;
  }
  while ( 2 )
  {
    --v26;
    v4 = &a1[v3 >> 5];
    sub_15B1350(
      (__int64)v33,
      *(unsigned __int64 **)(a1[1].m128i_i64[1] + 24),
      *(unsigned __int64 **)(a1[1].m128i_i64[1] + 32));
    v5 = v34;
    sub_15B1350(
      (__int64)v31,
      *(unsigned __int64 **)(v4->m128i_i64[1] + 24),
      *(unsigned __int64 **)(v4->m128i_i64[1] + 32));
    if ( v5 >= v32 )
    {
      sub_15B1350(
        (__int64)v33,
        *(unsigned __int64 **)(a1[1].m128i_i64[1] + 24),
        *(unsigned __int64 **)(a1[1].m128i_i64[1] + 32));
      v14 = v34;
      sub_15B1350(
        (__int64)v31,
        *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 24),
        *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 32));
      if ( v14 >= v32 )
      {
        sub_15B1350(
          (__int64)v33,
          *(unsigned __int64 **)(v4->m128i_i64[1] + 24),
          *(unsigned __int64 **)(v4->m128i_i64[1] + 32));
        v22 = v34;
        sub_15B1350(
          (__int64)v31,
          *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 24),
          *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 32));
        v23 = a1->m128i_i32[0];
        v24 = a1->m128i_i64[1];
        if ( v22 >= v32 )
        {
          *a1 = _mm_loadu_si128(v4);
          v4->m128i_i32[0] = v23;
          v4->m128i_i64[1] = v24;
        }
        else
        {
          *a1 = _mm_loadu_si128(v27 - 1);
          v27[-1].m128i_i32[0] = v23;
          v27[-1].m128i_i64[1] = v24;
        }
        v9 = a1[1].m128i_i64[1];
      }
      else
      {
        v15 = _mm_loadu_si128(a1 + 1);
        v9 = a1->m128i_i64[1];
        a1[1].m128i_i32[0] = a1->m128i_i32[0];
        a1[1].m128i_i64[1] = v9;
        *a1 = v15;
      }
    }
    else
    {
      sub_15B1350(
        (__int64)v33,
        *(unsigned __int64 **)(v4->m128i_i64[1] + 24),
        *(unsigned __int64 **)(v4->m128i_i64[1] + 32));
      v6 = v34;
      sub_15B1350(
        (__int64)v31,
        *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 24),
        *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 32));
      if ( v6 >= v32 )
      {
        sub_15B1350(
          (__int64)v33,
          *(unsigned __int64 **)(a1[1].m128i_i64[1] + 24),
          *(unsigned __int64 **)(a1[1].m128i_i64[1] + 32));
        v20 = v34;
        sub_15B1350(
          (__int64)v31,
          *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 24),
          *(unsigned __int64 **)(v27[-1].m128i_i64[1] + 32));
        v21 = a1->m128i_i32[0];
        v9 = a1->m128i_i64[1];
        if ( v20 >= v32 )
        {
          v25 = _mm_loadu_si128(a1 + 1);
          a1[1].m128i_i32[0] = v21;
          a1[1].m128i_i64[1] = v9;
          *a1 = v25;
        }
        else
        {
          *a1 = _mm_loadu_si128(v27 - 1);
          v27[-1].m128i_i32[0] = v21;
          v27[-1].m128i_i64[1] = v9;
          v9 = a1[1].m128i_i64[1];
        }
      }
      else
      {
        v7 = a1->m128i_i64[1];
        v8 = a1->m128i_i32[0];
        *a1 = _mm_loadu_si128(v4);
        v4->m128i_i32[0] = v8;
        v4->m128i_i64[1] = v7;
        v9 = a1[1].m128i_i64[1];
      }
    }
    v30 = a1 + 1;
    v10 = v27;
    while ( 1 )
    {
      v28 = (char *)v30;
      sub_15B1350((__int64)v33, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
      v29 = v34;
      sub_15B1350(
        (__int64)v31,
        *(unsigned __int64 **)(a1->m128i_i64[1] + 24),
        *(unsigned __int64 **)(a1->m128i_i64[1] + 32));
      if ( v29 < v32 )
        goto LABEL_7;
      do
      {
        --v10;
        sub_15B1350(
          (__int64)v33,
          *(unsigned __int64 **)(a1->m128i_i64[1] + 24),
          *(unsigned __int64 **)(a1->m128i_i64[1] + 32));
        v11 = v34;
        sub_15B1350(
          (__int64)v31,
          *(unsigned __int64 **)(v10->m128i_i64[1] + 24),
          *(unsigned __int64 **)(v10->m128i_i64[1] + 32));
      }
      while ( v11 < v32 );
      if ( v30 >= v10 )
        break;
      v12 = v30->m128i_i32[0];
      v13 = v30->m128i_i64[1];
      *v30 = _mm_loadu_si128(v10);
      v10->m128i_i32[0] = v12;
      v10->m128i_i64[1] = v13;
LABEL_7:
      v9 = v30[1].m128i_i64[1];
      ++v30;
    }
    v3 = (char *)v30 - (char *)a1;
    sub_3987CE0(v30, v27, v26);
    if ( (char *)v30 - (char *)a1 > 256 )
    {
      if ( v26 )
      {
        v27 = v30;
        continue;
      }
LABEL_18:
      v16 = v3 >> 4;
      v17 = (v16 - 2) >> 1;
      sub_3985B30((__int64)a1, v17, v16, a1[v17].m128i_i64[0], a1[v17].m128i_i64[1]);
      do
      {
        --v17;
        sub_3985B30((__int64)a1, v17, v16, a1[v17].m128i_i64[0], a1[v17].m128i_i64[1]);
      }
      while ( v17 );
      do
      {
        v28 -= 16;
        v18 = *(_QWORD *)v28;
        v19 = *((_QWORD *)v28 + 1);
        *(__m128i *)v28 = _mm_loadu_si128(a1);
        sub_3985B30((__int64)a1, 0, (v28 - (char *)a1) >> 4, v18, v19);
      }
      while ( v28 - (char *)a1 > 16 );
    }
    break;
  }
}
