// Function: sub_C22F50
// Address: 0xc22f50
//
__int64 __fastcall sub_C22F50(__int64 a1)
{
  __int64 result; // rax
  char v2; // r12
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rsi
  bool v7; // zf
  __m128i *v8; // rsi
  char *v9; // r8
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rax
  char *v14; // r15
  signed __int64 v15; // rdx
  _BYTE *v16; // rsi
  unsigned __int64 v17; // rax
  char *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-148h]
  char *v21; // [rsp+10h] [rbp-140h]
  const __m128i **v22; // [rsp+18h] [rbp-138h]
  __m128i v23; // [rsp+20h] [rbp-130h] BYREF
  __int64 v24; // [rsp+30h] [rbp-120h] BYREF
  unsigned __int64 v25; // [rsp+40h] [rbp-110h] BYREF
  char v26; // [rsp+50h] [rbp-100h]
  __int64 v27; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-E8h]
  char v29; // [rsp+70h] [rbp-E0h]
  __m128i v30; // [rsp+80h] [rbp-D0h] BYREF

  sub_C21E40((__int64)&v25, (_QWORD *)a1);
  if ( (v26 & 1) == 0 || (result = (unsigned int)v25, !(_DWORD)v25) )
  {
    v2 = *(_BYTE *)(a1 + 204);
    v22 = (const __m128i **)(a1 + 224);
    v3 = *(_QWORD *)(a1 + 224);
    if ( *(_QWORD *)(a1 + 232) != v3 )
      *(_QWORD *)(a1 + 232) = v3;
    sub_C22490(v22, v25);
    if ( *(_BYTE *)(a1 + 178) )
    {
      if ( v25 )
      {
LABEL_8:
        v4 = 0;
        while ( 1 )
        {
          sub_C20B00((__int64)&v27, a1);
          if ( (v29 & 1) != 0 )
          {
            result = (unsigned int)v27;
            v5 = v28;
            if ( (_DWORD)v27 )
              return result;
            v6 = v27;
            if ( !v2 )
            {
LABEL_19:
              v30.m128i_i64[0] = v6;
              v8 = *(__m128i **)(a1 + 232);
              v30.m128i_i64[1] = v5;
              if ( v8 == *(__m128i **)(a1 + 240) )
              {
                sub_C22DD0(v22, v8, &v30);
                goto LABEL_16;
              }
              if ( v8 )
              {
                *v8 = _mm_loadu_si128(&v30);
                v8 = *(__m128i **)(a1 + 232);
              }
              goto LABEL_15;
            }
          }
          else
          {
            v5 = v28;
            v6 = v27;
            if ( !v2 )
              goto LABEL_19;
          }
          v7 = *(_BYTE *)(a1 + 178) == 0;
          v23.m128i_i64[0] = v6;
          v23.m128i_i64[1] = v5;
          if ( v7 )
          {
            if ( v6 )
            {
              v20 = v5;
              sub_C7D030(&v30);
              sub_C7D280(&v30, v6, v20);
              sub_C7D290(&v30, &v24);
              v5 = v24;
            }
            v30.m128i_i64[0] = v5;
            v16 = *(_BYTE **)(a1 + 280);
            if ( v16 != *(_BYTE **)(a1 + 288) )
            {
              if ( v16 )
              {
                *(_QWORD *)v16 = v5;
                v16 = *(_BYTE **)(a1 + 280);
              }
              *(_QWORD *)(a1 + 280) = v16 + 8;
              v8 = *(__m128i **)(a1 + 232);
              if ( v8 == *(__m128i **)(a1 + 240) )
              {
LABEL_43:
                sub_C22C50(v22, v8, &v23);
                goto LABEL_16;
              }
              goto LABEL_13;
            }
            sub_A235E0(a1 + 272, v16, &v30);
          }
          v8 = *(__m128i **)(a1 + 232);
          if ( v8 == *(__m128i **)(a1 + 240) )
            goto LABEL_43;
LABEL_13:
          if ( v8 )
          {
            *v8 = _mm_loadu_si128(&v23);
            v8 = *(__m128i **)(a1 + 232);
          }
LABEL_15:
          *(_QWORD *)(a1 + 232) = v8 + 1;
LABEL_16:
          if ( v25 <= ++v4 )
            goto LABEL_33;
        }
      }
      goto LABEL_36;
    }
    v9 = *(char **)(a1 + 272);
    if ( *(char **)(a1 + 280) != v9 )
      *(_QWORD *)(a1 + 280) = v9;
    v10 = v25;
    if ( v2 )
    {
      if ( v25 > 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::reserve");
      if ( (__int64)(*(_QWORD *)(a1 + 288) - (_QWORD)v9) >> 3 < v25 )
      {
        v11 = 8 * v25;
        v12 = *(_QWORD *)(a1 + 280) - (_QWORD)v9;
        if ( v25 )
        {
          v13 = sub_22077B0(8 * v25);
          v9 = *(char **)(a1 + 272);
          v14 = (char *)v13;
          v15 = *(_QWORD *)(a1 + 280) - (_QWORD)v9;
        }
        else
        {
          v15 = *(_QWORD *)(a1 + 280) - (_QWORD)v9;
          v14 = 0;
        }
        if ( v15 > 0 )
        {
          v21 = v9;
          memmove(v14, v9, v15);
          v9 = v21;
          v19 = *(_QWORD *)(a1 + 288) - (_QWORD)v21;
        }
        else
        {
          if ( !v9 )
          {
LABEL_31:
            *(_QWORD *)(a1 + 272) = v14;
            *(_QWORD *)(a1 + 280) = &v14[v12];
            *(_QWORD *)(a1 + 288) = &v14[v11];
LABEL_32:
            if ( v25 )
              goto LABEL_8;
LABEL_33:
            if ( *(_BYTE *)(a1 + 178) )
            {
LABEL_36:
              sub_C1AFD0();
              return 0;
            }
            v9 = *(char **)(a1 + 272);
LABEL_35:
            *(_QWORD *)(a1 + 296) = v9;
            goto LABEL_36;
          }
          v19 = *(_QWORD *)(a1 + 288) - (_QWORD)v9;
        }
        j_j___libc_free_0(v9, v19);
        goto LABEL_31;
      }
    }
    else
    {
      v17 = (__int64)(*(_QWORD *)(a1 + 280) - (_QWORD)v9) >> 3;
      if ( v17 < v25 )
      {
        sub_C22AA0(a1 + 272, v25 - v17);
        goto LABEL_32;
      }
      if ( v17 > v25 )
      {
        v18 = &v9[8 * v25];
        if ( *(char **)(a1 + 280) != v18 )
          *(_QWORD *)(a1 + 280) = v18;
      }
    }
    if ( v10 )
      goto LABEL_8;
    goto LABEL_35;
  }
  return result;
}
