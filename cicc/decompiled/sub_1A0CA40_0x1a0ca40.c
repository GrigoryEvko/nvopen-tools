// Function: sub_1A0CA40
// Address: 0x1a0ca40
//
__int64 *__fastcall sub_1A0CA40(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v6; // rdx
  __int64 v7; // r11
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // r9
  __int64 v13; // rdi
  signed __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 *v26; // rcx
  __int64 v28; // r14
  __int64 v29; // r8
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rdx
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-B0h]
  signed __int64 v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+10h] [rbp-A0h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+20h] [rbp-90h]
  __int64 v45; // [rsp+20h] [rbp-90h]
  _QWORD *v46; // [rsp+20h] [rbp-90h]
  __int64 v47; // [rsp+28h] [rbp-88h]
  __int64 v48; // [rsp+30h] [rbp-80h]
  _QWORD *v49; // [rsp+30h] [rbp-80h]
  __int64 v50; // [rsp+38h] [rbp-78h]
  signed __int64 v51; // [rsp+38h] [rbp-78h]
  __int64 v52; // [rsp+40h] [rbp-70h] BYREF
  __int64 v53; // [rsp+48h] [rbp-68h]
  __int64 v54; // [rsp+50h] [rbp-60h]
  __int64 v55; // [rsp+58h] [rbp-58h]
  __int64 v56; // [rsp+60h] [rbp-50h] BYREF
  __int64 v57; // [rsp+68h] [rbp-48h]
  __int64 v58; // [rsp+70h] [rbp-40h]
  __int64 v59; // [rsp+78h] [rbp-38h]

  v6 = *a3;
  v7 = a3[2];
  v8 = a3[1];
  v9 = a3[3];
  v10 = v6 + 8;
  if ( v7 == v6 + 8 )
  {
    v10 = *(_QWORD *)(v9 + 8);
    v11 = v9 + 8;
    v40 = v10;
    v12 = v10 + 512;
  }
  else
  {
    v40 = v8;
    v11 = v9;
    v12 = v7;
  }
  v48 = a2[2];
  v47 = a2[4];
  v50 = a2[5];
  v13 = (v47 - v48) >> 3;
  v43 = a2[6];
  v44 = a2[7];
  v14 = v13 + ((((v9 - v50) >> 3) - 1) << 6) + ((v6 - v8) >> 3);
  v42 = a2[9];
  v41 = v14;
  if ( (unsigned __int64)(v13 + ((v43 - v44) >> 3) + ((((v42 - v50) >> 3) - 1) << 6)) >> 1 <= v14 )
  {
    if ( v43 != v10 )
    {
      if ( v11 == v42 )
      {
        v56 = v6;
        v57 = v8;
        v58 = v7;
        v59 = v9;
        sub_1A0C930(&v52, v10, v43, &v56);
      }
      else
      {
        v52 = v6;
        v53 = v8;
        v55 = v9;
        v28 = v11 + 8;
        v54 = v7;
        sub_1A0C930(&v56, v10, v12, &v52);
        v29 = v56;
        v30 = v57;
        v31 = v58;
        v32 = v59;
        if ( v42 != v28 )
        {
          v49 = a2;
          v33 = v28;
          do
          {
            v52 = v29;
            v33 += 8;
            v54 = v31;
            v55 = v32;
            v53 = v30;
            sub_1A0C930(&v56, *(_QWORD *)(v33 - 8), *(_QWORD *)(v33 - 8) + 512LL, &v52);
            v29 = v56;
            v30 = v57;
            v31 = v58;
            v32 = v59;
          }
          while ( v42 != v33 );
          a2 = v49;
        }
        v57 = v30;
        v59 = v32;
        v58 = v31;
        v56 = v29;
        sub_1A0C930(&v52, v44, v43, &v56);
      }
      v10 = a2[6];
      v44 = a2[7];
    }
    if ( v10 == v44 )
    {
      j_j___libc_free_0(v10, 512);
      v37 = (__int64 *)(a2[9] - 8LL);
      a2[9] = v37;
      v38 = *v37;
      v39 = *v37 + 512;
      a2[7] = v38;
      a2[8] = v39;
      a2[6] = v38 + 504;
    }
    else
    {
      a2[6] = v10 - 8;
    }
    v23 = a2[2];
    goto LABEL_14;
  }
  if ( v48 != v6 )
  {
    v45 = v9;
    if ( v50 == v9 )
    {
      v56 = v10;
      v58 = v12;
      v59 = v11;
      v57 = v40;
      sub_1A0C7C0(&v52, v48, v6, &v56);
    }
    else
    {
      v52 = v10;
      v55 = v11;
      v53 = v40;
      v54 = v12;
      sub_1A0C7C0(&v56, v8, v6, &v52);
      v15 = v56;
      v16 = v57;
      v17 = v58;
      v18 = v45 - 8;
      v19 = v50;
      v20 = v59;
      if ( v50 != v45 - 8 )
      {
        v51 = v14;
        v21 = v19;
        v46 = a2;
        v22 = v18;
        do
        {
          v52 = v15;
          v22 -= 8;
          v54 = v17;
          v55 = v20;
          v53 = v16;
          sub_1A0C7C0(&v56, *(_QWORD *)(v22 + 8), *(_QWORD *)(v22 + 8) + 512LL, &v52);
          v15 = v56;
          v16 = v57;
          v17 = v58;
          v20 = v59;
        }
        while ( v21 != v22 );
        v14 = v51;
        a2 = v46;
      }
      v57 = v16;
      v59 = v20;
      v58 = v17;
      v56 = v15;
      sub_1A0C7C0(&v52, v48, v47, &v56);
    }
    v6 = a2[2];
    v47 = a2[4];
  }
  if ( v6 != v47 - 8 )
  {
    v23 = a2[2] + 8LL;
    a2[2] = v23;
LABEL_14:
    v24 = a2[3];
    v25 = a2[4];
    v26 = (__int64 *)a2[5];
    v14 += (v23 - v24) >> 3;
    goto LABEL_15;
  }
  j_j___libc_free_0(a2[3], 512);
  v26 = (__int64 *)(a2[5] + 8LL);
  a2[5] = v26;
  v23 = *v26;
  v25 = *v26 + 512;
  a2[3] = *v26;
  v24 = v23;
  a2[4] = v25;
  a2[2] = v23;
LABEL_15:
  *a1 = v23;
  a1[1] = v24;
  a1[2] = v25;
  a1[3] = (__int64)v26;
  if ( v14 < 0 )
  {
    v34 = ~((unsigned __int64)~v14 >> 6);
    goto LABEL_31;
  }
  if ( v14 > 63 )
  {
    v34 = v14 >> 6;
LABEL_31:
    v35 = &v26[v34];
    a1[3] = (__int64)v35;
    v36 = *v35;
    a1[1] = v36;
    a1[2] = v36 + 512;
    *a1 = v36 + 8 * (v14 - (v34 << 6));
    return a1;
  }
  *a1 = v23 + 8 * v41;
  return a1;
}
