// Function: sub_1606530
// Address: 0x1606530
//
__int64 __fastcall sub_1606530(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // rbx
  __int64 v9; // r15
  _QWORD *v10; // rdi
  _QWORD *v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 i; // r13
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // [rsp-98h] [rbp-98h]
  __int64 v25; // [rsp-90h] [rbp-90h]
  _QWORD *v26; // [rsp-80h] [rbp-80h]
  __int64 v27; // [rsp-70h] [rbp-70h] BYREF
  __int64 v28; // [rsp-68h] [rbp-68h]
  __int64 v29; // [rsp-50h] [rbp-50h] BYREF
  __int64 v30; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
    return result;
  v3 = sub_16982B0(a1, a2);
  v6 = sub_16982C0(a1, a2, v4, v5);
  v7 = v6;
  if ( v3 == v6 )
  {
    sub_169C630(&v27, v6, 1);
    sub_169C630(&v29, v7, 2);
  }
  else
  {
    sub_1699170(&v27, v3, 1);
    sub_1699170(&v29, v3, 2);
  }
  v8 = *(_QWORD **)(a1 + 8);
  result = (__int64)&v8[5 * *(unsigned int *)(a1 + 24)];
  v26 = (_QWORD *)result;
  if ( v8 != (_QWORD *)result )
  {
    while ( 1 )
    {
      result = v8[1];
      if ( result == v27 )
      {
        v10 = v8 + 1;
        if ( v7 == result )
          result = sub_169CB90(v10, &v27);
        else
          result = sub_1698510(v10, &v27);
        if ( (_BYTE)result )
          goto LABEL_11;
        result = v8[1];
        if ( v29 == result )
          goto LABEL_19;
LABEL_7:
        v9 = v8[4];
        if ( v9 )
        {
          if ( v7 == *(_QWORD *)(v9 + 32) )
          {
            v21 = *(_QWORD *)(v9 + 40);
            if ( v21 )
            {
              v22 = 32LL * *(_QWORD *)(v21 - 8);
              v23 = v21 + v22;
              if ( v21 != v21 + v22 )
              {
                do
                {
                  v24 = v21;
                  v25 = v23 - 32;
                  sub_127D120((_QWORD *)(v23 - 24));
                  v23 = v25;
                  v21 = v24;
                }
                while ( v24 != v25 );
              }
              j_j_j___libc_free_0_0(v21 - 8);
            }
          }
          else
          {
            sub_1698460(v9 + 32);
          }
          sub_164BE60(v9);
          result = sub_1648B90(v9);
        }
LABEL_11:
        if ( v7 == v8[1] )
          goto LABEL_23;
LABEL_12:
        result = sub_1698460(v8 + 1);
LABEL_13:
        v8 += 5;
        if ( v26 == v8 )
          break;
      }
      else
      {
        if ( v29 != result )
          goto LABEL_7;
LABEL_19:
        v11 = v8 + 1;
        if ( v7 == result )
          result = sub_169CB90(v11, &v29);
        else
          result = sub_1698510(v11, &v29);
        if ( !(_BYTE)result )
          goto LABEL_7;
        if ( v7 != v8[1] )
          goto LABEL_12;
LABEL_23:
        v12 = v8[2];
        if ( !v12 )
          goto LABEL_13;
        v13 = 32LL * *(_QWORD *)(v12 - 8);
        for ( i = v12 + v13; v12 != i; sub_127D120((_QWORD *)(i + 8)) )
          i -= 32;
        v8 += 5;
        result = j_j_j___libc_free_0_0(v12 - 8);
        if ( v26 == v8 )
          break;
      }
    }
  }
  if ( v7 == v29 )
  {
    v15 = v30;
    if ( v30 )
    {
      v16 = 32LL * *(_QWORD *)(v30 - 8);
      v17 = v30 + v16;
      if ( v30 != v30 + v16 )
      {
        do
        {
          v17 -= 32;
          sub_127D120((_QWORD *)(v17 + 8));
        }
        while ( v15 != v17 );
      }
      result = j_j_j___libc_free_0_0(v15 - 8);
    }
  }
  else
  {
    result = sub_1698460(&v29);
  }
  if ( v7 != v27 )
    return sub_1698460(&v27);
  v18 = v28;
  if ( v28 )
  {
    v19 = 32LL * *(_QWORD *)(v28 - 8);
    v20 = v28 + v19;
    if ( v28 != v28 + v19 )
    {
      do
      {
        v20 -= 32;
        sub_127D120((_QWORD *)(v20 + 8));
      }
      while ( v18 != v20 );
    }
    return j_j_j___libc_free_0_0(v18 - 8);
  }
  return result;
}
