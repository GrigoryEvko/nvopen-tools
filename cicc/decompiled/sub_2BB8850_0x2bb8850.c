// Function: sub_2BB8850
// Address: 0x2bb8850
//
__int64 __fastcall sub_2BB8850(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 *v6; // r15
  __int64 v8; // r12
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // r14
  unsigned __int64 *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r13
  unsigned __int64 *v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v29; // [rsp+0h] [rbp-40h]

  v6 = (unsigned __int64 *)a3;
  v8 = a5;
  v9 = a1;
  if ( a1 != a2 && a3 != a4 )
  {
    do
    {
      if ( (unsigned __int8)sub_2B1D420(*(unsigned __int8 **)(*v6 + 8), *(unsigned __int8 **)(*v9 + 8), a3, a4, a5, a6) )
      {
        v10 = v6;
        v11 = v8;
        v6 += 8;
        v8 += 64;
        sub_2BB7BD0(v11, v10, v12, v13, v14, v15);
        if ( v9 == a2 )
          break;
      }
      else
      {
        v16 = v9;
        v17 = v8;
        v9 += 8;
        v8 += 64;
        sub_2BB7BD0(v17, v16, v12, v13, v14, v15);
        if ( v9 == a2 )
          break;
      }
    }
    while ( v6 != (unsigned __int64 *)a4 );
  }
  v29 = (char *)a2 - (char *)v9;
  v18 = ((char *)a2 - (char *)v9) >> 6;
  if ( (char *)a2 - (char *)v9 > 0 )
  {
    v19 = v8;
    do
    {
      v20 = v9;
      v21 = v19;
      v9 += 8;
      v19 += 64;
      sub_2BB7BD0(v21, v20, a3, a4, a5, a6);
      --v18;
    }
    while ( v18 );
    a3 = v29;
    if ( v29 <= 0 )
      a3 = 64;
    v8 += a3;
  }
  v22 = a4 - (_QWORD)v6;
  v23 = (a4 - (__int64)v6) >> 6;
  if ( a4 - (__int64)v6 > 0 )
  {
    v24 = v8;
    do
    {
      v25 = v6;
      v26 = v24;
      v6 += 8;
      v24 += 64;
      sub_2BB7BD0(v26, v25, a3, a4, a5, a6);
      --v23;
    }
    while ( v23 );
    v27 = 64;
    if ( v22 > 0 )
      v27 = v22;
    v8 += v27;
  }
  return v8;
}
