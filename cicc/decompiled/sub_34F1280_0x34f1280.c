// Function: sub_34F1280
// Address: 0x34f1280
//
__int64 __fastcall sub_34F1280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // r10
  __int64 v10; // r15
  unsigned int v11; // r14d
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned int v14; // edx
  __m128i v15; // xmm1
  char v17; // al
  __int64 v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+10h] [rbp-80h]
  __int8 v20; // [rsp+1Fh] [rbp-71h]
  __int64 v21; // [rsp+20h] [rbp-70h]
  __int64 v22; // [rsp+30h] [rbp-60h]
  __int64 v23; // [rsp+38h] [rbp-58h]
  __m128i v24; // [rsp+40h] [rbp-50h] BYREF

  v19 = a4 + 8 * a5;
  v24 = 0;
  if ( v19 == a4 )
  {
    v20 = 0;
    v18 = 0;
    goto LABEL_21;
  }
  v20 = 0;
  v7 = a4;
  v18 = 0;
  do
  {
    v8 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
    if ( v8 == v8 + 40LL * (*(_DWORD *)(*(_QWORD *)v7 + 40LL) & 0xFFFFFF) )
      goto LABEL_20;
    v21 = v7;
    v9 = a2;
    v10 = v8 + 40LL * (*(_DWORD *)(*(_QWORD *)v7 + 40LL) & 0xFFFFFF);
    while ( 2 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v8 )
        {
          v11 = *(_DWORD *)(v8 + 8);
          if ( v11 )
          {
            v12 = *(_QWORD *)(a3 + 32);
            v13 = v12 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF);
            if ( v12 != v13 )
              break;
          }
        }
        v8 += 40;
        if ( v10 == v8 )
          goto LABEL_26;
      }
      do
      {
        while ( 1 )
        {
          if ( !*(_BYTE *)v12 )
          {
            v14 = *(_DWORD *)(v12 + 8);
            if ( v14 )
            {
              if ( v11 == v14 )
                goto LABEL_16;
              if ( v11 - 1 <= 0x3FFFFFFE && v14 - 1 <= 0x3FFFFFFE )
                break;
            }
          }
LABEL_12:
          v12 += 40;
          if ( v13 == v12 )
            goto LABEL_25;
        }
        v22 = a3;
        v23 = v9;
        v17 = sub_E92070(*(_QWORD *)(v9 + 208), v11, v14);
        v9 = v23;
        a3 = v22;
        if ( v17 )
        {
LABEL_16:
          if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 || (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
          {
            v7 = v21;
            a2 = v9;
            if ( !v20 )
            {
              v18 = v21;
              v20 = 1;
              goto LABEL_20;
            }
            *(_BYTE *)a1 = 0;
            *(_BYTE *)(a1 + 16) = 0;
            return a1;
          }
          goto LABEL_12;
        }
        v12 += 40;
      }
      while ( v13 != v12 );
LABEL_25:
      v8 += 40;
      if ( v10 != v8 )
        continue;
      break;
    }
LABEL_26:
    v7 = v21;
    a2 = v9;
LABEL_20:
    v7 += 8;
  }
  while ( v7 != v19 );
LABEL_21:
  v24.m128i_i64[0] = v18;
  v24.m128i_i8[8] = v20;
  v15 = _mm_loadu_si128(&v24);
  *(_BYTE *)a1 = 1;
  *(__m128i *)(a1 + 8) = v15;
  return a1;
}
