// Function: sub_28FB170
// Address: 0x28fb170
//
__int64 *__fastcall sub_28FB170(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 *v4; // r12
  _QWORD *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r10
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // r14
  unsigned __int64 v12; // rcx
  signed __int64 v13; // rbx
  __int64 v14; // rax
  _QWORD *v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 *v24; // rcx
  __int64 v26; // r13
  _QWORD *v27; // r9
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // r12
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rcx
  __int64 v37; // rdi
  __int64 v38; // rdx
  signed __int64 v39; // [rsp+0h] [rbp-B0h]
  __int64 v40; // [rsp+8h] [rbp-A8h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  __int64 v43; // [rsp+20h] [rbp-90h]
  __int64 v44; // [rsp+28h] [rbp-88h]
  __int64 v45; // [rsp+30h] [rbp-80h]
  __int64 v46; // [rsp+30h] [rbp-80h]
  __int64 v47; // [rsp+38h] [rbp-78h]
  signed __int64 v48; // [rsp+38h] [rbp-78h]
  _QWORD *v49; // [rsp+40h] [rbp-70h] BYREF
  __int64 v50; // [rsp+48h] [rbp-68h]
  __int64 v51; // [rsp+50h] [rbp-60h]
  __int64 v52; // [rsp+58h] [rbp-58h]
  _QWORD *v53; // [rsp+60h] [rbp-50h] BYREF
  __int64 v54; // [rsp+68h] [rbp-48h]
  __int64 v55; // [rsp+70h] [rbp-40h]
  __int64 v56; // [rsp+78h] [rbp-38h]

  v4 = a1;
  v5 = *(_QWORD **)a3;
  v6 = *(_QWORD *)(a3 + 16);
  v7 = *(_QWORD *)(a3 + 8);
  v8 = *(_QWORD *)(a3 + 24);
  v9 = *(_QWORD *)a3 + 24LL;
  if ( v6 == v9 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    v10 = v8 + 8;
    v11 = v9;
    v41 = v9 + 504;
  }
  else
  {
    v41 = *(_QWORD *)(a3 + 16);
    v10 = *(_QWORD *)(a3 + 24);
    v11 = *(_QWORD *)(a3 + 8);
  }
  v47 = a2[5];
  v42 = a2[4];
  v44 = a2[2];
  v12 = 0xAAAAAAAAAAAAAAABLL * ((v42 - v44) >> 3);
  v43 = a2[6];
  v45 = a2[7];
  v13 = v12 + 21 * (((v8 - v47) >> 3) - 1) - 0x5555555555555555LL * (((__int64)v5 - v7) >> 3);
  v40 = a2[9];
  v39 = v13;
  if ( (v12 + 21 * (((v40 - v47) >> 3) - 1) - 0x5555555555555555LL * ((v43 - v45) >> 3)) >> 1 <= v13 )
  {
    if ( v43 != v9 )
    {
      if ( v10 == v40 )
      {
        v53 = *(_QWORD **)a3;
        v54 = v7;
        v55 = v6;
        v56 = v8;
        sub_28FAF30(&v49, v9, v43, &v53);
      }
      else
      {
        v49 = *(_QWORD **)a3;
        v51 = v6;
        v26 = v10 + 8;
        v50 = v7;
        v52 = v8;
        sub_28FAF30(&v53, v9, v41, &v49);
        v27 = v53;
        v28 = v54;
        v29 = v55;
        v30 = v56;
        if ( v40 != v26 )
        {
          v31 = v26;
          do
          {
            v49 = v27;
            v31 += 8;
            v51 = v29;
            v52 = v30;
            v50 = v28;
            sub_28FAF30(&v53, *(_QWORD *)(v31 - 8), *(_QWORD *)(v31 - 8) + 504LL, &v49);
            v27 = v53;
            v28 = v54;
            v29 = v55;
            v30 = v56;
          }
          while ( v40 != v31 );
          v4 = a1;
        }
        v54 = v28;
        v56 = v30;
        v55 = v29;
        v53 = v27;
        sub_28FAF30(&v49, v45, v43, &v53);
      }
      v9 = a2[6];
      v45 = a2[7];
    }
    v32 = (_QWORD *)(v9 - 24);
    if ( v9 == v45 )
    {
      j_j___libc_free_0(v9);
      v36 = (__int64 *)(a2[9] - 8LL);
      a2[9] = v36;
      v37 = *v36;
      v38 = *v36 + 504;
      a2[7] = *v36;
      v32 = (_QWORD *)(v37 + 480);
      a2[8] = v38;
    }
    a2[6] = v32;
    sub_D68D70(v32);
    v21 = a2[2];
    goto LABEL_14;
  }
  if ( (_QWORD *)v44 != v5 )
  {
    v46 = *(_QWORD *)(a3 + 24);
    if ( v47 == v8 )
    {
      v53 = (_QWORD *)v9;
      v54 = v11;
      v56 = v10;
      v55 = v41;
      sub_28FACB0(&v49, v44, (__int64)v5, &v53);
    }
    else
    {
      v50 = v11;
      v49 = (_QWORD *)v9;
      v51 = v41;
      v52 = v10;
      sub_28FACB0(&v53, v7, (__int64)v5, &v49);
      v14 = v47;
      v15 = v53;
      v16 = v54;
      v17 = v55;
      v18 = v56;
      if ( v47 != v46 - 8 )
      {
        v48 = v13;
        v19 = v14;
        v20 = v46 - 8;
        do
        {
          v49 = v15;
          v20 -= 8;
          v51 = v17;
          v52 = v18;
          v50 = v16;
          sub_28FACB0(&v53, *(_QWORD *)(v20 + 8), *(_QWORD *)(v20 + 8) + 504LL, &v49);
          v15 = v53;
          v16 = v54;
          v17 = v55;
          v18 = v56;
        }
        while ( v19 != v20 );
        v13 = v48;
        v4 = a1;
      }
      v54 = v16;
      v56 = v18;
      v55 = v17;
      v53 = v15;
      sub_28FACB0(&v49, v44, v42, &v53);
    }
    v5 = (_QWORD *)a2[2];
    v42 = a2[4];
  }
  if ( v5 != (_QWORD *)(v42 - 24) )
  {
    sub_D68D70(v5);
    v21 = a2[2] + 24LL;
    a2[2] = v21;
LABEL_14:
    v22 = a2[3];
    v23 = a2[4];
    v24 = (__int64 *)a2[5];
    v13 += 0xAAAAAAAAAAAAAAABLL * ((v21 - v22) >> 3);
    goto LABEL_15;
  }
  sub_D68D70(v5);
  j_j___libc_free_0(a2[3]);
  v24 = (__int64 *)(a2[5] + 8LL);
  a2[5] = v24;
  v21 = *v24;
  v23 = *v24 + 504;
  a2[3] = *v24;
  v22 = v21;
  a2[4] = v23;
  a2[2] = v21;
LABEL_15:
  *v4 = v21;
  v4[1] = v22;
  v4[2] = v23;
  v4[3] = (__int64)v24;
  if ( v13 < 0 )
  {
    v33 = ~(~v13 / 0x15uLL);
    goto LABEL_31;
  }
  if ( v13 > 20 )
  {
    v33 = v13 / 21;
LABEL_31:
    v34 = &v24[v33];
    v4[3] = (__int64)v34;
    v35 = *v34;
    v4[1] = v35;
    v4[2] = v35 + 504;
    *v4 = v35 + 24 * (v13 - 21 * v33);
    return v4;
  }
  *v4 = v21 + 24 * v39;
  return v4;
}
