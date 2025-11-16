// Function: sub_30C89C0
// Address: 0x30c89c0
//
__int64 *__fastcall sub_30C89C0(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // r8
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v13; // rcx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // r10
  unsigned __int64 *v20; // rbx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned __int64 *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-80h]
  __int64 v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  unsigned __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  unsigned __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h]
  __int64 v39; // [rsp+40h] [rbp-40h]
  _QWORD *v40; // [rsp+48h] [rbp-38h]

  v7 = a4 - a3;
  v8 = a4 - a3;
  v9 = a1[2];
  v10 = v8 >> 3;
  if ( *a2 != v9 )
  {
    v11 = a1[6];
    if ( *a2 != v11 )
    {
      v37 = *a2;
      v38 = a2[1];
      v39 = a2[2];
      v40 = (_QWORD *)a2[3];
      return sub_30C7D10(a1, &v37, a3, a4, v10);
    }
    v13 = a1[8];
    v14 = ((v13 - v11) >> 3) - 1;
    if ( v10 > v14 )
    {
      v33 = v10;
      sub_30C7760(a1, v10 - v14);
      v11 = a1[6];
      v13 = a1[8];
      v10 = v33;
    }
    v15 = a1[7];
    v16 = a1[9];
    v17 = ((__int64)(v11 - v15) >> 3) + v10;
    if ( v17 < 0 )
    {
      v22 = ~((unsigned __int64)~v17 >> 6);
    }
    else
    {
      if ( v17 <= 63 )
      {
        v18 = v11 + v7;
        v19 = v13;
        v20 = (unsigned __int64 *)a1[9];
        v21 = a1[7];
        v32 = v18;
LABEL_9:
        v37 = v11;
        v38 = v15;
        v39 = v13;
        v40 = (_QWORD *)v16;
        v30 = v19;
        sub_30C5F60(v36, a3, a4, &v37);
        a1[7] = v21;
        a1[9] = (unsigned __int64)v20;
        a1[6] = v32;
        a1[8] = v30;
        return (__int64 *)v32;
      }
      v22 = v17 >> 6;
    }
    v20 = (unsigned __int64 *)(v16 + 8 * v22);
    v21 = *v20;
    v19 = *v20 + 512;
    v32 = *v20 + 8 * (v17 - (v22 << 6));
    goto LABEL_9;
  }
  v23 = a1[3];
  v24 = (__int64)(v9 - v23) >> 3;
  if ( v7 > v9 - v23 )
  {
    v35 = v10;
    sub_30C7690(a1, v10 - v24);
    v9 = a1[2];
    v23 = a1[3];
    v10 = v35;
    v24 = (__int64)(v9 - v23) >> 3;
  }
  v25 = a1[4];
  v26 = (unsigned __int64 *)a1[5];
  v27 = v24 - v10;
  if ( v27 < 0 )
  {
    v29 = ~((unsigned __int64)~v27 >> 6);
    goto LABEL_20;
  }
  if ( v27 > 63 )
  {
    v29 = v27 >> 6;
LABEL_20:
    v26 += v29;
    v23 = *v26;
    v25 = *v26 + 512;
    v28 = *v26 + 8 * (v27 - (v29 << 6));
    goto LABEL_17;
  }
  v28 = v9 - 8 * v10;
LABEL_17:
  v37 = v28;
  v31 = v28;
  v38 = v23;
  v39 = v25;
  v34 = v25;
  v40 = v26;
  sub_30C5F60(v36, a3, a4, &v37);
  a1[3] = v23;
  a1[5] = (unsigned __int64)v26;
  a1[2] = v31;
  a1[4] = v34;
  return (__int64 *)v31;
}
