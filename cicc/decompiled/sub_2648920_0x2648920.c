// Function: sub_2648920
// Address: 0x2648920
//
char *__fastcall sub_2648920(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 a8)
{
  char *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r8
  __int64 v15; // r10
  __int64 *v16; // rax
  int v17; // ecx
  char *result; // rax
  __int64 *v19; // r14
  __int64 *v20; // rbx
  __int64 *i; // r12
  __int64 v22; // rsi
  __int64 v23; // rax
  volatile signed __int32 *v24; // rdi
  __int64 v25; // rcx
  volatile signed __int32 *v26; // rdi
  __int64 *v27; // r12
  __int64 *v28; // r13
  __int64 v29; // r14
  __int64 *v30; // rbx
  __int64 v31; // rcx
  volatile signed __int32 *v32; // rdi
  __int64 v33; // rcx
  volatile signed __int32 *v34; // rdi
  __int64 v35; // rax
  char *v36; // rbx
  __int64 v37; // r12
  char *v38; // r13
  __int64 v39; // rdx
  volatile signed __int32 *v40; // rdi
  __int64 v41; // [rsp+8h] [rbp-88h]
  __int64 v42; // [rsp+8h] [rbp-88h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  __int64 v47; // [rsp+18h] [rbp-78h]
  __int64 v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  __int64 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+20h] [rbp-70h]
  __int64 v53; // [rsp+20h] [rbp-70h]
  char *v54; // [rsp+20h] [rbp-70h]
  __int64 v55; // [rsp+20h] [rbp-70h]
  char *v56; // [rsp+20h] [rbp-70h]
  __int64 v57; // [rsp+20h] [rbp-70h]
  __int64 *v58; // [rsp+28h] [rbp-68h]
  char *v59; // [rsp+30h] [rbp-60h]
  __int64 v60; // [rsp+38h] [rbp-58h]
  __int64 *v61; // [rsp+38h] [rbp-58h]
  __int64 *v62; // [rsp+40h] [rbp-50h]
  __int64 *v63; // [rsp+48h] [rbp-48h]
  _QWORD v64[7]; // [rsp+58h] [rbp-38h] BYREF

  while ( 1 )
  {
    v8 = a6;
    v62 = a1;
    v63 = (__int64 *)a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( a4 <= v9 )
      break;
    v10 = a5;
    if ( a5 <= a7 )
    {
      result = sub_2643F00(a2, a3, a6);
      v64[0] = a8;
      if ( a1 == a2 )
        return (char *)sub_2643DC0((__int64)v8, result, v63);
      if ( v8 != result )
      {
        v19 = a2 - 2;
        v20 = (__int64 *)(result - 16);
        for ( i = v63 - 2; ; i -= 2 )
        {
          if ( sub_2648420(v64, v20, (__int64)v19) )
          {
            v22 = *v19;
            v23 = v19[1];
            *v19 = 0;
            v19[1] = 0;
            v24 = (volatile signed __int32 *)i[1];
            *i = v22;
            i[1] = v23;
            if ( v24 )
              sub_A191D0(v24);
            if ( v19 == v62 )
            {
              result = (char *)i;
              v36 = (char *)(v20 + 2);
              v37 = (v36 - v8) >> 4;
              if ( v36 - v8 > 0 )
              {
                v38 = result;
                do
                {
                  v38 -= 16;
                  v39 = *((_QWORD *)v36 - 2);
                  result = (char *)*((_QWORD *)v36 - 1);
                  v36 -= 16;
                  *((_QWORD *)v36 + 1) = 0;
                  *(_QWORD *)v36 = 0;
                  v40 = (volatile signed __int32 *)*((_QWORD *)v38 + 1);
                  *(_QWORD *)v38 = v39;
                  *((_QWORD *)v38 + 1) = result;
                  if ( v40 )
                    result = (char *)sub_A191D0(v40);
                  --v37;
                }
                while ( v37 );
              }
              return result;
            }
            v19 -= 2;
          }
          else
          {
            v25 = *v20;
            result = (char *)v20[1];
            *v20 = 0;
            v20[1] = 0;
            v26 = (volatile signed __int32 *)i[1];
            *i = v25;
            i[1] = (__int64)result;
            if ( v26 )
              result = (char *)sub_A191D0(v26);
            if ( v8 == (char *)v20 )
              return result;
            v20 -= 2;
          }
        }
      }
      return result;
    }
    if ( a4 <= a5 )
    {
      v49 = a4;
      v55 = a5 / 2;
      v58 = &a2[2 * (a5 / 2)];
      v35 = sub_26485A0((__int64)a1, (__int64)a2, v58, a8);
      v13 = a7;
      v12 = v49;
      v59 = (char *)v35;
      v14 = v55;
      v60 = (v35 - (__int64)a1) >> 4;
    }
    else
    {
      v52 = a4;
      v60 = a4 / 2;
      v59 = (char *)&a1[2 * (a4 / 2)];
      v11 = sub_26486A0(a2, a3, (__int64)v59, a8);
      v12 = v52;
      v13 = a7;
      v58 = v11;
      v14 = ((char *)v11 - (char *)a2) >> 4;
    }
    v15 = v12 - v60;
    if ( v15 <= v14 || v14 > v13 )
    {
      if ( v15 > v13 )
      {
        v46 = v13;
        v51 = v14;
        v57 = v15;
        v16 = (__int64 *)sub_263E6E0(v59, (char *)a2, (char *)v58);
        v13 = v46;
        v14 = v51;
        v15 = v57;
      }
      else
      {
        v16 = v58;
        if ( v15 )
        {
          v44 = v14;
          v41 = v13;
          v48 = v15;
          v54 = sub_2643F00((__int64 *)v59, (__int64)a2, v8);
          sub_2643F00(a2, (__int64)v58, v59);
          v16 = sub_2643DC0((__int64)v8, v54, v58);
          v15 = v48;
          v14 = v44;
          v13 = v41;
        }
      }
    }
    else
    {
      v16 = (__int64 *)v59;
      if ( v14 )
      {
        v45 = v14;
        v42 = v13;
        v50 = v15;
        v56 = sub_2643F00(a2, (__int64)v58, v8);
        sub_2643DC0((__int64)v59, a2, v58);
        v16 = (__int64 *)sub_2643F00((__int64 *)v8, (__int64)v56, v59);
        v15 = v50;
        v14 = v45;
        v13 = v42;
      }
    }
    v17 = v60;
    v43 = v15;
    v47 = v13;
    v53 = v14;
    v61 = v16;
    sub_2648920((_DWORD)a1, (_DWORD)v59, (_DWORD)v16, v17, v14, (_DWORD)v8, v13, a8);
    a6 = v8;
    a2 = v58;
    a7 = v47;
    a3 = (__int64)v63;
    a5 = v10 - v53;
    a4 = v43;
    a1 = v61;
  }
  v27 = a2;
  result = sub_2643F00(a1, (__int64)a2, a6);
  v64[0] = a8;
  v28 = a1;
  v29 = (__int64)result;
  if ( result == v8 )
    return result;
  v30 = (__int64 *)v8;
  do
  {
    while ( 1 )
    {
      if ( v63 == v27 )
        return sub_2643F00(v30, v29, v28);
      if ( sub_2648420(v64, v27, (__int64)v30) )
        break;
      v33 = *v30;
      result = (char *)v30[1];
      *v30 = 0;
      v30[1] = 0;
      v34 = (volatile signed __int32 *)v28[1];
      *v28 = v33;
      v28[1] = (__int64)result;
      if ( v34 )
        result = (char *)sub_A191D0(v34);
      v30 += 2;
      v28 += 2;
      if ( (__int64 *)v29 == v30 )
        return result;
    }
    v31 = *v27;
    result = (char *)v27[1];
    *v27 = 0;
    v27[1] = 0;
    v32 = (volatile signed __int32 *)v28[1];
    *v28 = v31;
    v28[1] = (__int64)result;
    if ( v32 )
      result = (char *)sub_A191D0(v32);
    v27 += 2;
    v28 += 2;
  }
  while ( (__int64 *)v29 != v30 );
  return result;
}
