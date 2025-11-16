// Function: sub_272ECC0
// Address: 0x272ecc0
//
void __fastcall sub_272ECC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rdi
  signed __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rbx
  __int64 v25; // rdi
  signed __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-60h]
  signed __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]

  v5 = a3;
  v6 = a2 - a1;
  v7 = (a2 - a1) >> 3;
  v8 = 0xCF3CF3CF3CF3CF3DLL * v7;
  v32 = v5 + a2 - a1;
  v31 = 0xCF3CF3CF3CF3CF3DLL * v7;
  if ( a2 - a1 <= 1008 )
  {
    sub_272DD00(a1, a2, v7, v8, a5, v5);
  }
  else
  {
    v9 = a1;
    do
    {
      v10 = v9;
      v9 += 1176;
      sub_272DD00(v10, v9, v7, v8, a5, v5);
    }
    while ( a2 - v9 > 1008 );
    sub_272DD00(v9, a2, v7, v8, a5, v5);
    if ( v6 > 1176 )
    {
      v12 = 7;
      while ( 1 )
      {
        v13 = 2 * v12;
        if ( v31 < 2 * v12 )
        {
          v14 = a3;
          v19 = v31;
          v15 = a1;
        }
        else
        {
          v14 = a3;
          v15 = a1;
          v27 = v12;
          v16 = 336 * v12;
          v17 = 168 * v12;
          do
          {
            v18 = v15;
            v15 += v16;
            v14 = sub_272EAA0(v18, v15 + v17 - v16, v15 + v17 - v16, v15, v14, v11);
            v19 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - v15) >> 3);
          }
          while ( v13 <= v19 );
          v12 = v27;
        }
        if ( v12 <= v19 )
          v19 = v12;
        v12 *= 4;
        sub_272EAA0(v15, v15 + 168 * v19, v15 + 168 * v19, a2, v14, v11);
        v21 = a1;
        if ( v31 < v12 )
          break;
        v28 = v13;
        v22 = 168 * v12;
        v23 = a3;
        v24 = 168 * v13;
        do
        {
          v25 = v23;
          v23 += v22;
          v21 = sub_272E8B0(v25, v23 + v24 - v22, v23 + v24 - v22, v23, v21, v20);
          v26 = 0xCF3CF3CF3CF3CF3DLL * ((v32 - v23) >> 3);
        }
        while ( v12 <= v26 );
        if ( v26 > v28 )
          v26 = v28;
        sub_272E8B0(v23, v23 + 168 * v26, v23 + 168 * v26, v32, v21, v20);
        if ( v31 <= v12 )
          return;
      }
      if ( v31 <= v13 )
        v13 = v31;
      sub_272E8B0(a3, a3 + 168 * v13, a3 + 168 * v13, v32, a1, v20);
    }
  }
}
