// Function: sub_173D940
// Address: 0x173d940
//
__int64 __fastcall sub_173D940(__int64 a1, __int64 a2, _DWORD *a3, unsigned int a4)
{
  __int16 *v5; // r14
  __int16 *v6; // rax
  __int64 *v7; // rsi
  __int16 *v9; // r12
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]

  v5 = *(__int16 **)(a2 + 8);
  v6 = (__int16 *)sub_16982C0();
  v7 = (__int64 *)(a2 + 8);
  if ( v5 == v6 )
  {
    v9 = v6;
    sub_16A2F10(&v25, (__int64)v7, a3, a4);
    sub_169C7E0(&v27, &v25);
    sub_169C7E0((_QWORD *)(a1 + 8), &v27);
    v10 = v28;
    if ( v28 )
    {
      v11 = 32LL * *(_QWORD *)(v28 - 8);
      v12 = v28 + v11;
      if ( v28 != v28 + v11 )
      {
        do
        {
          v12 -= 32;
          if ( v9 == *(__int16 **)(v12 + 8) )
          {
            v13 = *(_QWORD *)(v12 + 16);
            if ( v13 )
            {
              v14 = 32LL * *(_QWORD *)(v13 - 8);
              v15 = v13 + v14;
              if ( v13 != v13 + v14 )
              {
                do
                {
                  v15 -= 32;
                  v23 = v13;
                  sub_127D120((_QWORD *)(v15 + 8));
                  v13 = v23;
                }
                while ( v23 != v15 );
              }
              j_j_j___libc_free_0_0(v13 - 8);
            }
          }
          else
          {
            sub_1698460(v12 + 8);
          }
        }
        while ( v10 != v12 );
      }
      j_j_j___libc_free_0_0(v10 - 8);
    }
    v16 = v26;
    if ( v26 )
    {
      v17 = 32LL * *(_QWORD *)(v26 - 8);
      v18 = v26 + v17;
      if ( v26 != v26 + v17 )
      {
        do
        {
          v18 -= 32;
          if ( v9 == *(__int16 **)(v18 + 8) )
          {
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 )
            {
              v20 = 32LL * *(_QWORD *)(v19 - 8);
              v21 = v19 + v20;
              if ( v19 != v19 + v20 )
              {
                do
                {
                  v21 -= 32;
                  v24 = v19;
                  sub_127D120((_QWORD *)(v21 + 8));
                  v19 = v24;
                }
                while ( v24 != v21 );
              }
              j_j_j___libc_free_0_0(v19 - 8);
            }
          }
          else
          {
            sub_1698460(v18 + 8);
          }
        }
        while ( v16 != v18 );
      }
      j_j_j___libc_free_0_0(v16 - 8);
    }
  }
  else
  {
    sub_169C410(&v25, v7, a3, a4);
    sub_1698450((__int64)&v27, (__int64)&v25);
    sub_169E320((_QWORD *)(a1 + 8), &v27, v5);
    sub_1698460((__int64)&v27);
    sub_1698460((__int64)&v25);
  }
  return a1;
}
