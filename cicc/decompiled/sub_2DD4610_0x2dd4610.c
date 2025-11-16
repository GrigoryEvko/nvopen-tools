// Function: sub_2DD4610
// Address: 0x2dd4610
//
void __fastcall sub_2DD4610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // r12
  __int64 v20; // rdi
  signed __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // r14
  __int64 v25; // r15
  __int64 v26; // rbx
  __int64 v27; // rdi
  signed __int64 v28; // rax
  __int64 v29; // [rsp+0h] [rbp-60h]
  signed __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v8 = a2 - a1;
  v9 = (a2 - a1) >> 4;
  v10 = 0xCCCCCCCCCCCCCCCDLL * v9;
  v35 = a3 + a2 - a1;
  v33 = 0xCCCCCCCCCCCCCCCDLL * v9;
  if ( a2 - a1 <= 480 )
  {
    sub_2DD38C0(a1, a2, v9, v10, a5, a6);
  }
  else
  {
    v11 = a1;
    do
    {
      v12 = v11;
      v11 += 560;
      sub_2DD38C0(v12, v11, v9, v10, a5, a6);
    }
    while ( a2 - v11 > 480 );
    sub_2DD38C0(v11, a2, v9, v10, a5, a6);
    if ( v8 > 560 )
    {
      v34 = a2;
      v14 = 7;
      while ( 1 )
      {
        v15 = 2 * v14;
        if ( v33 < 2 * v14 )
        {
          v16 = a3;
          v21 = v33;
          v17 = a1;
        }
        else
        {
          v16 = a3;
          v17 = a1;
          v29 = v14;
          v18 = 160 * v14;
          v19 = 80 * v14;
          do
          {
            v20 = v17;
            v17 += v18;
            v16 = sub_2DD43F0(v20, v17 + v19 - v18, v17 + v19 - v18, v17, v16, v13);
            v21 = 0xCCCCCCCCCCCCCCCDLL * ((v34 - v17) >> 4);
          }
          while ( v15 <= v21 );
          v14 = v29;
        }
        if ( v14 <= v21 )
          v21 = v14;
        v14 *= 4;
        sub_2DD43F0(v17, v17 + 80 * v21, v17 + 80 * v21, v34, v16, v13);
        v23 = a1;
        if ( v33 < v14 )
          break;
        v30 = v15;
        v24 = a3;
        v25 = 80 * v14;
        v26 = 80 * v15;
        do
        {
          v27 = v24;
          v24 += v25;
          v23 = sub_2DD41F0(v27, v24 + v26 - v25, v24 + v26 - v25, v24, v23, v22);
          v28 = 0xCCCCCCCCCCCCCCCDLL * ((v35 - v24) >> 4);
        }
        while ( v14 <= v28 );
        if ( v28 > v30 )
          v28 = v30;
        sub_2DD41F0(v24, v24 + 80 * v28, v24 + 80 * v28, v35, v23, v22);
        if ( v33 <= v14 )
          return;
      }
      if ( v33 <= v15 )
        v15 = v33;
      sub_2DD41F0(a3, a3 + 80 * v15, a3 + 80 * v15, v35, a1, v22);
    }
  }
}
