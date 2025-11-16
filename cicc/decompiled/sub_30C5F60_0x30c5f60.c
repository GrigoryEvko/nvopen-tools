// Function: sub_30C5F60
// Address: 0x30c5f60
//
__int64 *__fastcall sub_30C5F60(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rdx
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r11
  __int64 v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx

  v4 = a3 - a2;
  v7 = v4 >> 3;
  v8 = *a4;
  v9 = a4[1];
  v10 = a4[2];
  v11 = (__int64 *)a4[3];
  if ( v4 > 0 )
  {
    v12 = a2;
    while ( 1 )
    {
      v13 = v12;
      v14 = (v10 - v8) >> 3;
      if ( v14 > v7 )
        v14 = v7;
      v12 += 8 * v14;
      if ( 8 * v14 > 0 )
      {
        v15 = 0;
        do
        {
          *(_QWORD *)(v8 + 8 * v15) = *(_QWORD *)(v13 + 8 * v15);
          ++v15;
        }
        while ( ((8 * v14) >> 3) - v15 > 0 );
      }
      v16 = v14 + ((v8 - v9) >> 3);
      if ( v16 < 0 )
        break;
      v8 += 8 * v14;
      if ( v16 > 63 )
      {
        v17 = v16 >> 6;
LABEL_11:
        v11 += v17;
        v9 = *v11;
        v10 = *v11 + 512;
        v8 = *v11 + 8 * (v16 - (v17 << 6));
      }
      v7 -= v14;
      if ( v7 <= 0 )
        goto LABEL_13;
    }
    v17 = ~((unsigned __int64)~v16 >> 6);
    goto LABEL_11;
  }
LABEL_13:
  a1[2] = v10;
  a1[3] = (__int64)v11;
  *a1 = v8;
  a1[1] = v9;
  return a1;
}
