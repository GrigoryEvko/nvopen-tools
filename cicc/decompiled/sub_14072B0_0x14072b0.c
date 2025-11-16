// Function: sub_14072B0
// Address: 0x14072b0
//
__int64 *__fastcall sub_14072B0(__int64 *a1, _QWORD *a2, __int64 *a3, __int64 *a4)
{
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v9; // r11
  __int64 *v10; // r13
  __int64 v11; // r10
  _QWORD *v12; // rax
  __int64 v13; // r14
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 *v19; // r12
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 *i; // r14
  __int64 v25; // rdi
  unsigned __int64 v26; // r12
  __int64 *v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v32; // rcx
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // [rsp+0h] [rbp-100h]
  __int64 v39; // [rsp+8h] [rbp-F8h]
  _QWORD *v40; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v41; // [rsp+18h] [rbp-E8h]
  __int64 v42; // [rsp+20h] [rbp-E0h]
  __int64 v43; // [rsp+28h] [rbp-D8h]
  __int64 v44; // [rsp+30h] [rbp-D0h]
  _QWORD *v45; // [rsp+38h] [rbp-C8h]
  _QWORD *v46; // [rsp+38h] [rbp-C8h]
  __int64 v47; // [rsp+40h] [rbp-C0h]
  __int64 v48; // [rsp+40h] [rbp-C0h]
  __int64 v49; // [rsp+48h] [rbp-B8h]
  __int64 *v50; // [rsp+48h] [rbp-B8h]
  __int64 v51[4]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+70h] [rbp-90h] BYREF
  __int64 v53; // [rsp+78h] [rbp-88h]
  __int64 v54; // [rsp+80h] [rbp-80h]
  __int64 *v55; // [rsp+88h] [rbp-78h]
  __int64 v56; // [rsp+90h] [rbp-70h] BYREF
  __int64 v57; // [rsp+98h] [rbp-68h]
  __int64 v58; // [rsp+A0h] [rbp-60h]
  _QWORD *v59; // [rsp+A8h] [rbp-58h]
  __m128i v60; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v61; // [rsp+C0h] [rbp-40h]
  _QWORD *v62; // [rsp+C8h] [rbp-38h]

  v5 = *a3;
  v6 = *a4;
  if ( *a3 == *a4 )
  {
    v30 = a3[1];
    *a1 = v5;
    a1[1] = v30;
    a1[2] = a3[2];
    a1[3] = a3[3];
    return a1;
  }
  v9 = a2[2];
  v10 = (__int64 *)a2[5];
  v11 = a2[6];
  v47 = a2[3];
  v49 = a2[4];
  v12 = (_QWORD *)a2[9];
  v45 = v12;
  if ( v5 != v9 || v6 != v11 )
  {
    v40 = (_QWORD *)a4[3];
    v43 = a4[1];
    v39 = a3[2];
    v38 = a3[3];
    v13 = (((((__int64)v40 - v38) >> 3) - 1) << 6) + ((v6 - v43) >> 3) + ((v39 - v5) >> 3);
    v14 = (v49 - v9) >> 3;
    v44 = a3[1];
    v41 = ((v5 - v44) >> 3) + ((((v38 - (__int64)v10) >> 3) - 1) << 6) + v14;
    v42 = a2[7];
    v15 = (v11 - v42) >> 3;
    if ( (unsigned __int64)(v14 + v15 + ((v12 - v10 - 1) << 6) - v13) >> 1 >= v41 )
    {
      if ( v5 != v9 )
      {
        v59 = (_QWORD *)a3[3];
        v16 = a4[2];
        v56 = v5;
        v62 = v40;
        v58 = v39;
        v60.m128i_i64[1] = v43;
        v61 = v16;
        v57 = v44;
        v52 = v9;
        v53 = v47;
        v55 = v10;
        v60.m128i_i64[0] = v6;
        v54 = v49;
        sub_1405B90(v51, &v52, (__int64)&v56, &v60);
        v9 = a2[2];
        v10 = (__int64 *)a2[5];
        v47 = a2[3];
        v49 = a2[4];
      }
      v17 = v13 + ((v9 - v47) >> 3);
      if ( v17 < 0 )
      {
        v32 = ~((unsigned __int64)~v17 >> 6);
      }
      else
      {
        if ( v17 <= 63 )
        {
          v18 = v9 + 8 * v13;
          v19 = v10;
LABEL_9:
          a2[2] = v18;
          a2[5] = v19;
          a2[3] = v47;
          a2[4] = v49;
          goto LABEL_10;
        }
        v32 = v17 >> 6;
      }
      v19 = &v10[v32];
      v47 = *v19;
      v18 = *v19 + 8 * (v17 - (v32 << 6));
      v49 = *v19 + 512;
      while ( v10 < v19 )
      {
        v33 = *v10++;
        j_j___libc_free_0(v33, 512);
      }
      goto LABEL_9;
    }
    v21 = a2[8];
    if ( v6 != v11 )
    {
      v22 = a4[2];
      v60.m128i_i64[0] = *a3;
      v56 = v11;
      v54 = v22;
      v60.m128i_i64[1] = v44;
      v62 = (_QWORD *)v38;
      v61 = v39;
      v59 = v45;
      v57 = v42;
      v53 = v43;
      v58 = v21;
      v55 = v40;
      v52 = v6;
      sub_1405E20(v51, (__int64)&v52, &v56, (__int64)&v60);
      v11 = a2[6];
      v21 = a2[8];
      v42 = a2[7];
      v45 = (_QWORD *)a2[9];
      v15 = (v11 - v42) >> 3;
    }
    v23 = v15 - v13;
    if ( v23 < 0 )
    {
      v34 = ~((unsigned __int64)~v23 >> 6);
    }
    else
    {
      if ( v23 <= 63 )
      {
        v48 = v11 - 8 * v13;
        v50 = v45;
LABEL_18:
        for ( i = v50 + 1; v45 + 1 > i; ++i )
        {
          v25 = *i;
          j_j___libc_free_0(v25, 512);
        }
        v18 = a2[2];
        a2[8] = v21;
        v19 = (__int64 *)a2[5];
        a2[6] = v48;
        a2[7] = v42;
        a2[9] = v50;
        v47 = a2[3];
        v49 = a2[4];
LABEL_10:
        *a1 = v18;
        a1[3] = (__int64)v19;
        a1[2] = v49;
        a1[1] = v47;
        v20 = v41 + ((v18 - v47) >> 3);
        if ( v20 < 0 )
        {
          v35 = ~((unsigned __int64)~v20 >> 6);
        }
        else
        {
          if ( v20 <= 63 )
          {
            *a1 = v18 + 8 * v41;
            return a1;
          }
          v35 = v20 >> 6;
        }
        v36 = &v19[v35];
        a1[3] = (__int64)v36;
        v37 = *v36;
        a1[1] = v37;
        a1[2] = v37 + 512;
        *a1 = v37 + 8 * (v20 - (v35 << 6));
        return a1;
      }
      v34 = v23 >> 6;
    }
    v50 = &v45[v34];
    v42 = *v50;
    v21 = *v50 + 512;
    v48 = *v50 + 8 * (v23 - (v34 << 6));
    goto LABEL_18;
  }
  v26 = (unsigned __int64)(v12 + 1);
  if ( v12 + 1 > v10 + 1 )
  {
    v46 = a2;
    v27 = v10 + 1;
    v28 = v9;
    do
    {
      v29 = *v27++;
      j_j___libc_free_0(v29, 512);
    }
    while ( v26 > (unsigned __int64)v27 );
    a2 = v46;
    v9 = v28;
  }
  a2[6] = v9;
  a2[9] = v10;
  a2[7] = v47;
  a2[8] = v49;
  *a1 = v9;
  a1[1] = v47;
  a1[2] = v49;
  a1[3] = (__int64)v10;
  return a1;
}
