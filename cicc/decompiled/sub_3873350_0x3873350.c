// Function: sub_3873350
// Address: 0x3873350
//
__int64 __fastcall sub_3873350(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 *v10; // r13
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 *v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // r14
  __int64 *v20; // r12
  __int64 *v21; // r14
  __int64 *v22; // rax
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 *v25; // r10
  __int64 v26; // rbx
  __int64 *i; // r11
  __int64 *v28; // rax
  int v29; // r11d
  __int64 v30; // r8
  char *v31; // r10
  __int64 *v32; // r13
  __int64 v33; // r12
  char *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rdx
  __int64 v37; // r14
  __int64 v38; // rcx
  __int64 *v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rsi
  __int64 *v42; // r14
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r14
  __int64 v46; // rdx
  __int64 *v47; // rax
  __int64 *v48; // r14
  __int64 *v49; // rbx
  __int64 v50; // rdx
  int v52; // [rsp+10h] [rbp-60h]
  __int64 *v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  int v55; // [rsp+18h] [rbp-58h]
  __int64 *v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  __int64 *v58; // [rsp+20h] [rbp-50h]
  __int64 *v59; // [rsp+20h] [rbp-50h]
  __int64 *v60; // [rsp+28h] [rbp-48h]
  __int64 v61[7]; // [rsp+38h] [rbp-38h] BYREF

  result = a5;
  v10 = a1;
  v11 = a2;
  v12 = (__int64 *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_23:
    v37 = (char *)v11 - (char *)v10;
    v38 = ((char *)v11 - (char *)v10) >> 4;
    if ( (char *)v11 - (char *)v10 > 0 )
    {
      v39 = a6;
      v40 = v10;
      do
      {
        v41 = *v40;
        v39 += 2;
        v40 += 2;
        *(v39 - 2) = v41;
        *(v39 - 1) = *(v40 - 1);
        --v38;
      }
      while ( v38 );
      if ( v37 <= 0 )
        v37 = 16;
      result = a8;
      v42 = (__int64 *)((char *)a6 + v37);
      v61[0] = a8;
      if ( v12 == v11 )
      {
LABEL_38:
        if ( a6 != v42 )
        {
          v45 = (char *)v42 - (char *)a6;
          result = v45 >> 4;
          if ( v45 > 0 )
          {
            do
            {
              v46 = *a6;
              v10 += 2;
              a6 += 2;
              *(v10 - 2) = v46;
              *(v10 - 1) = *(a6 - 1);
              --result;
            }
            while ( result );
          }
        }
      }
      else if ( a6 != v42 )
      {
        while ( 1 )
        {
          if ( sub_386ECF0(v61, *v11, v11[1], *a6, a6[1]) )
          {
            v43 = *v11;
            v10 += 2;
            v11 += 2;
            *(v10 - 2) = v43;
            result = *(v11 - 1);
            *(v10 - 1) = result;
            if ( a6 == v42 )
              return result;
          }
          else
          {
            v44 = *a6;
            a6 += 2;
            v10 += 2;
            *(v10 - 2) = v44;
            result = *(a6 - 1);
            *(v10 - 1) = result;
            if ( a6 == v42 )
              return result;
          }
          if ( v12 == v11 )
            goto LABEL_38;
        }
      }
    }
  }
  else
  {
    v13 = a5;
    if ( a7 < a5 )
    {
      v25 = a2;
      v26 = a4;
      for ( i = a1; ; i = v58 )
      {
        if ( v13 < v26 )
        {
          v55 = (int)i;
          v59 = v25;
          v33 = v26 / 2;
          v32 = &i[2 * (v26 / 2)];
          v47 = sub_386F500(v25, a3, v32, a8);
          v31 = (char *)v59;
          v29 = v55;
          v60 = v47;
          v30 = ((char *)v47 - (char *)v59) >> 4;
        }
        else
        {
          v53 = v25;
          v56 = i;
          v60 = &v25[2 * (v13 / 2)];
          v28 = sub_386F3E0(i, (__int64)v25, v60, a8);
          v29 = (int)v56;
          v30 = v13 / 2;
          v31 = (char *)v53;
          v32 = v28;
          v33 = ((char *)v28 - (char *)v56) >> 4;
        }
        v26 -= v33;
        v52 = v29;
        v57 = v30;
        v34 = sub_3873120((char *)v32, v31, (__int64)v60, v26, v30, a6, a7);
        v35 = v57;
        v58 = (__int64 *)v34;
        v54 = v35;
        sub_3873350(v52, (_DWORD)v32, (_DWORD)v34, v33, v35, (_DWORD)a6, a7, a8);
        result = (__int64)v58;
        v13 -= v54;
        v36 = v13;
        if ( v13 > a7 )
          v36 = a7;
        if ( v26 <= v36 )
        {
          v12 = (__int64 *)a3;
          v11 = v60;
          v10 = v58;
          goto LABEL_23;
        }
        if ( v13 <= a7 )
          break;
        v25 = v60;
      }
      v12 = (__int64 *)a3;
      v11 = v60;
      v10 = v58;
    }
    v14 = (char *)v12 - (char *)v11;
    v15 = ((char *)v12 - (char *)v11) >> 4;
    if ( (char *)v12 - (char *)v11 > 0 )
    {
      v16 = a6;
      v17 = v11;
      do
      {
        v18 = *v17;
        v16 += 2;
        v17 += 2;
        *(v16 - 2) = v18;
        *(v16 - 1) = *(v17 - 1);
        --v15;
      }
      while ( v15 );
      if ( v14 <= 0 )
        v14 = 16;
      result = a8;
      v19 = (__int64 *)((char *)a6 + v14);
      v61[0] = a8;
      if ( v10 == v11 )
      {
        result = v14 >> 4;
        while ( 1 )
        {
          v19 -= 2;
          *(v12 - 2) = v18;
          v12 -= 2;
          v12[1] = v19[1];
          if ( !--result )
            break;
          v18 = *(v19 - 2);
        }
      }
      else if ( a6 != v19 )
      {
        v20 = v11 - 2;
        v21 = v19 - 2;
        v22 = v12;
        v23 = v10;
        v24 = v22;
        while ( 1 )
        {
          while ( 1 )
          {
            v24 -= 2;
            if ( sub_386ECF0(v61, *v21, v21[1], *v20, v20[1]) )
              break;
            *v24 = *v21;
            result = v21[1];
            v24[1] = result;
            if ( a6 == v21 )
              return result;
            v21 -= 2;
          }
          *v24 = *v20;
          v24[1] = v20[1];
          if ( v23 == v20 )
            break;
          v20 -= 2;
        }
        v48 = v21 + 2;
        v49 = v24;
        result = ((char *)v48 - (char *)a6) >> 4;
        if ( (char *)v48 - (char *)a6 > 0 )
        {
          do
          {
            v50 = *(v48 - 2);
            v48 -= 2;
            v49 -= 2;
            *v49 = v50;
            v49[1] = v48[1];
            --result;
          }
          while ( result );
        }
      }
    }
  }
  return result;
}
