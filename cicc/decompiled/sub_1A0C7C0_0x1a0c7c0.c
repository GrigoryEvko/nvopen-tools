// Function: sub_1A0C7C0
// Address: 0x1a0c7c0
//
__int64 *__fastcall sub_1A0C7C0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r10
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 i; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax

  v6 = a3 - a2;
  v7 = a3;
  v8 = v6 >> 3;
  v9 = *a4;
  v10 = a4[1];
  if ( v6 > 0 )
  {
    v11 = (v9 - v10) >> 3;
    if ( v9 != v10 )
      goto LABEL_14;
LABEL_3:
    v12 = 64;
    for ( i = *(_QWORD *)(a4[3] - 8) + 512LL; ; i = v9 )
    {
      if ( v12 > v8 )
        v12 = v8;
      v14 = v7 - 8 * v12;
      v15 = (8 * v12) >> 3;
      if ( 8 * v12 > 0 )
      {
        v16 = -8 * v15 + v7;
        v17 = i - 8 * v15;
        do
        {
          *(_QWORD *)(v17 + 8 * v15 - 8) = *(_QWORD *)(v16 + 8 * v15 - 8);
          --v15;
        }
        while ( v15 );
        v9 = *a4;
        v10 = a4[1];
        v11 = (*a4 - v10) >> 3;
      }
      v18 = v11 - v12;
      if ( v18 < 0 )
      {
        v19 = ~((unsigned __int64)~v18 >> 6);
      }
      else
      {
        if ( v18 <= 63 )
        {
          v8 -= v12;
          v9 -= 8 * v12;
          *a4 = v9;
          if ( v8 <= 0 )
            break;
          goto LABEL_13;
        }
        v19 = v18 >> 6;
      }
      v8 -= v12;
      v20 = (__int64 *)(a4[3] + 8 * v19);
      a4[3] = (__int64)v20;
      v10 = *v20;
      v21 = *v20 + 512;
      v9 = v10 + 8 * (v18 - (v19 << 6));
      a4[1] = v10;
      a4[2] = v21;
      *a4 = v9;
      if ( v8 <= 0 )
        break;
LABEL_13:
      v7 = v14;
      v11 = (v9 - v10) >> 3;
      if ( v9 == v10 )
        goto LABEL_3;
LABEL_14:
      v12 = v11;
    }
  }
  v22 = a4[2];
  *a1 = v9;
  a1[1] = v10;
  a1[2] = v22;
  a1[3] = a4[3];
  return a1;
}
