// Function: sub_1540D70
// Address: 0x1540d70
//
__int64 __fastcall sub_1540D70(
        __int64 a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v8; // r15
  __int64 result; // rax
  __int64 **v10; // r13
  __int64 *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 **v15; // rbx
  __int64 v16; // r12
  __int64 **v17; // r14
  __int64 *v18; // rax
  __int64 *v19; // rax
  char *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r10
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 *v25; // rdx
  char *v26; // rax
  __int64 v27; // rsi
  __int64 *v28; // rbx
  char *v29; // r14
  char *v30; // rbx
  __int64 **i; // r13
  bool v32; // al
  __int64 **v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r10
  __int64 v37; // r8
  __int64 v38; // r11
  __int64 v39; // r12
  char *v40; // r14
  __int64 v41; // rbx
  __int64 **v42; // r15
  __int64 v43; // rdx
  __int64 *v44; // rcx
  __int64 v45; // rax
  char *v46; // rbx
  __int64 *v47; // rcx
  __int64 **v48; // [rsp+0h] [rbp-80h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 **v53; // [rsp+20h] [rbp-60h]
  __int64 v54; // [rsp+28h] [rbp-58h]
  __int64 **v55; // [rsp+38h] [rbp-48h]
  _QWORD v56[7]; // [rsp+48h] [rbp-38h] BYREF

  while ( 1 )
  {
    v8 = a6;
    v55 = (__int64 **)a3;
    result = a7;
    if ( a5 <= a7 )
      result = a5;
    if ( a4 <= result )
      break;
    v20 = a2;
    v21 = a5;
    if ( a5 <= a7 )
    {
      v22 = a3;
      v23 = a3 - (_QWORD)a2;
      v24 = (a3 - (__int64)a2) >> 4;
      if ( a3 - (__int64)a2 <= 0 )
        return result;
      v25 = a6;
      v26 = a2;
      do
      {
        v27 = *(_QWORD *)v26;
        v25 += 2;
        v26 += 16;
        *(v25 - 2) = v27;
        *((_DWORD *)v25 - 2) = *((_DWORD *)v26 - 2);
        --v24;
      }
      while ( v24 );
      result = 16;
      v56[0] = a8;
      if ( v23 <= 0 )
        v23 = 16;
      v28 = (__int64 *)((char *)a6 + v23);
      if ( v20 == (char *)a1 )
      {
        result = v23 >> 4;
        while ( 1 )
        {
          v28 -= 2;
          *(_QWORD *)(v22 - 16) = v27;
          v22 -= 16;
          *(_DWORD *)(v22 + 8) = *((_DWORD *)v28 + 2);
          if ( !--result )
            break;
          v27 = *(v28 - 2);
        }
        return result;
      }
      if ( a6 == v28 )
        return result;
      v29 = v20 - 16;
      v30 = (char *)(v28 - 2);
      for ( i = v55 - 2; ; i -= 2 )
      {
        v32 = sub_153CA80((__int64)v56, (__int64 **)v30, (__int64 **)v29);
        v33 = i;
        if ( v32 )
        {
          *i = *(__int64 **)v29;
          *((_DWORD *)i + 2) = *((_DWORD *)v29 + 2);
          if ( v29 == (char *)a1 )
          {
            v46 = v30 + 16;
            result = (v46 - (char *)v8) >> 4;
            if ( v46 - (char *)v8 > 0 )
            {
              do
              {
                v47 = (__int64 *)*((_QWORD *)v46 - 2);
                v46 -= 16;
                v33 -= 2;
                *v33 = v47;
                *((_DWORD *)v33 + 2) = *((_DWORD *)v46 + 2);
                --result;
              }
              while ( result );
            }
            return result;
          }
          v29 -= 16;
        }
        else
        {
          *i = *(__int64 **)v30;
          result = *((unsigned int *)v30 + 2);
          *((_DWORD *)i + 2) = result;
          if ( v8 == (__int64 *)v30 )
            return result;
          v30 -= 16;
        }
      }
    }
    v34 = a4;
    if ( a4 > a5 )
    {
      v54 = a4 / 2;
      v48 = (__int64 **)(a1 + 16 * (a4 / 2));
      v45 = sub_153CE20((__int64)a2, a3, v48, a8);
      v38 = a8;
      v36 = a7;
      v53 = (__int64 **)v45;
      v37 = (v45 - (__int64)a2) >> 4;
    }
    else
    {
      v50 = a5 / 2;
      v53 = (__int64 **)&a2[16 * (a5 / 2)];
      v35 = sub_153CEB0(a1, (__int64)a2, v53, a8);
      v36 = a7;
      v37 = v50;
      v48 = (__int64 **)v35;
      v38 = a8;
      v54 = (v35 - a1) >> 4;
    }
    v39 = v34 - v54;
    v49 = v38;
    v51 = v36;
    v52 = v37;
    v40 = sub_1540B50((char *)v48, a2, (__int64)v53, v39, v37, v8, v36);
    sub_1540D70(a1, (_DWORD)v48, (_DWORD)v40, v54, v52, (_DWORD)v8, v51, v49);
    a6 = v8;
    a3 = (__int64)v55;
    a4 = v39;
    a8 = v49;
    a2 = (char *)v53;
    a1 = (__int64)v40;
    a7 = v51;
    a5 = v21 - v52;
  }
  v10 = (__int64 **)a2;
  v11 = a6;
  result = a1;
  v12 = (__int64)&a2[-a1];
  v13 = (__int64)&a2[-a1] >> 4;
  if ( v12 > 0 )
  {
    do
    {
      v14 = *(_QWORD *)result;
      v11 += 2;
      result += 16;
      *(v11 - 2) = v14;
      *((_DWORD *)v11 - 2) = *(_DWORD *)(result - 8);
      --v13;
    }
    while ( v13 );
    result = 16;
    v56[0] = a8;
    v15 = (__int64 **)((char *)a6 + v12);
    if ( a6 != (__int64 *)v15 )
    {
      v16 = a1;
      v17 = (__int64 **)a6;
      while ( 1 )
      {
        result = v16;
        if ( v10 == v55 )
          break;
        if ( sub_153CA80((__int64)v56, v10, v17) )
        {
          v18 = *v10;
          v16 += 16;
          v10 += 2;
          *(_QWORD *)(v16 - 16) = v18;
          result = *((unsigned int *)v10 - 2);
          *(_DWORD *)(v16 - 8) = result;
          if ( v17 == v15 )
            return result;
        }
        else
        {
          v19 = *v17;
          v17 += 2;
          v16 += 16;
          *(_QWORD *)(v16 - 16) = v19;
          result = *((unsigned int *)v17 - 2);
          *(_DWORD *)(v16 - 8) = result;
          if ( v17 == v15 )
            return result;
        }
      }
      v41 = (char *)v15 - (char *)v17;
      v42 = v17;
      v43 = v41 >> 4;
      if ( v41 > 0 )
      {
        do
        {
          v44 = *v42;
          result += 16;
          v42 += 2;
          *(_QWORD *)(result - 16) = v44;
          *(_DWORD *)(result - 8) = *((_DWORD *)v42 - 2);
          --v43;
        }
        while ( v43 );
      }
    }
  }
  return result;
}
