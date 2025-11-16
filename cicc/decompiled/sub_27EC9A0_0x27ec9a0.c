// Function: sub_27EC9A0
// Address: 0x27ec9a0
//
char __fastcall sub_27EC9A0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 *a8,
        __int64 *a9,
        __int64 *a10)
{
  __int64 *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdi
  char result; // al
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 *v43; // [rsp+10h] [rbp-90h]
  __m128i v44; // [rsp+20h] [rbp-80h] BYREF
  __int64 v45; // [rsp+30h] [rbp-70h]
  __int64 v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+40h] [rbp-60h]
  __int64 v48; // [rsp+48h] [rbp-58h]
  __int64 v49; // [rsp+50h] [rbp-50h]
  __int64 v50; // [rsp+58h] [rbp-48h]
  __int16 v51; // [rsp+60h] [rbp-40h]

  v10 = a1;
  v11 = (char *)a2 - (char *)a1;
  v12 = v11 >> 5;
  if ( v11 >> 7 > 0 )
  {
    v43 = &a1[16 * (v11 >> 7)];
    while ( 1 )
    {
      v25 = *a9;
      v26 = *a8;
      v27 = *a10;
      v28 = *v10;
      v45 = 0;
      v46 = v26;
      v47 = v25;
      v44 = (__m128i)a7;
      v48 = v27;
      v49 = 0;
      v50 = 0;
      v51 = 257;
      if ( !(unsigned __int8)sub_9AC470(v28, &v44, 0) )
        return a2 == v10;
      v13 = *a9;
      v14 = *a8;
      v15 = *a10;
      v45 = 0;
      v16 = v10[4];
      v46 = v14;
      v47 = v13;
      v44 = (__m128i)a7;
      v48 = v15;
      v49 = 0;
      v50 = 0;
      v51 = 257;
      if ( !(unsigned __int8)sub_9AC470(v16, &v44, 0) )
        return a2 == v10 + 4;
      v17 = *a9;
      v18 = *a8;
      v19 = *a10;
      v45 = 0;
      v20 = v10[8];
      v46 = v18;
      v47 = v17;
      v44 = (__m128i)a7;
      v48 = v19;
      v49 = 0;
      v50 = 0;
      v51 = 257;
      if ( !(unsigned __int8)sub_9AC470(v20, &v44, 0) )
        return a2 == v10 + 8;
      v21 = *a8;
      v22 = *a9;
      v23 = *a10;
      v45 = 0;
      v46 = v21;
      v24 = v10[12];
      v47 = v22;
      v51 = 257;
      v44 = (__m128i)a7;
      v48 = v23;
      v49 = 0;
      v50 = 0;
      if ( !(unsigned __int8)sub_9AC470(v24, &v44, 0) )
        return a2 == v10 + 12;
      v10 += 16;
      if ( v10 == v43 )
      {
        v12 = ((char *)a2 - (char *)v10) >> 5;
        break;
      }
    }
  }
  if ( v12 == 2 )
    goto LABEL_17;
  if ( v12 == 3 )
  {
    v30 = *a9;
    v31 = *a8;
    v32 = *a10;
    v45 = 0;
    v33 = *v10;
    v46 = v31;
    v44 = (__m128i)a7;
    v47 = v30;
    v48 = v32;
    v49 = 0;
    v50 = 0;
    v51 = 257;
    if ( !(unsigned __int8)sub_9AC470(v33, &v44, 0) )
      return a2 == v10;
    v10 += 4;
LABEL_17:
    v34 = *a9;
    v35 = *a8;
    v36 = *a10;
    v45 = 0;
    v47 = v34;
    v37 = *v10;
    v46 = v35;
    v51 = 257;
    v44 = (__m128i)a7;
    v48 = v36;
    v49 = 0;
    v50 = 0;
    if ( (unsigned __int8)sub_9AC470(v37, &v44, 0) )
    {
      v10 += 4;
      goto LABEL_19;
    }
    return a2 == v10;
  }
  if ( v12 != 1 )
    return 1;
LABEL_19:
  v38 = *a10;
  v39 = *a9;
  v40 = *a8;
  v45 = 0;
  v41 = *v10;
  v47 = v39;
  v44 = (__m128i)a7;
  v48 = v38;
  v46 = v40;
  v49 = 0;
  v50 = 0;
  v51 = 257;
  result = sub_9AC470(v41, &v44, 0);
  if ( !result )
    return a2 == v10;
  return result;
}
