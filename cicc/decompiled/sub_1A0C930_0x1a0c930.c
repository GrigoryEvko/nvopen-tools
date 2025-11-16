// Function: sub_1A0C930
// Address: 0x1a0c930
//
_QWORD *__fastcall sub_1A0C930(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rdx
  __int64 v7; // r11
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rsi

  v4 = a3 - a2;
  v7 = v4 >> 3;
  v8 = a4[2];
  v9 = *a4;
  if ( v4 > 0 )
  {
    v10 = a2;
    while ( 1 )
    {
      v11 = v10;
      v12 = (v8 - v9) >> 3;
      if ( v12 > v7 )
        v12 = v7;
      v13 = 8 * v12;
      v14 = v12;
      v10 += 8 * v12;
      v15 = (8 * v12) >> 3;
      if ( 8 * v12 > 0 )
      {
        v16 = 0;
        do
        {
          *(_QWORD *)(v9 + 8 * v16) = *(_QWORD *)(v11 + 8 * v16);
          ++v16;
        }
        while ( v15 - v16 > 0 );
        v9 = *a4;
      }
      v17 = v14 + ((v9 - a4[1]) >> 3);
      if ( v17 < 0 )
        break;
      if ( v17 > 63 )
      {
        v21 = v17 >> 6;
LABEL_15:
        v22 = (__int64 *)(a4[3] + 8 * v21);
        a4[3] = (__int64)v22;
        v23 = *v22;
        v8 = *v22 + 512;
        v9 = v23 + 8 * (v17 - (v21 << 6));
        a4[1] = v23;
        a4[2] = v8;
        *a4 = v9;
        goto LABEL_12;
      }
      v9 += v13;
      v8 = a4[2];
      *a4 = v9;
LABEL_12:
      v7 -= v14;
      if ( v7 <= 0 )
        goto LABEL_13;
    }
    v21 = ~((unsigned __int64)~v17 >> 6);
    goto LABEL_15;
  }
LABEL_13:
  v18 = a4[1];
  a1[2] = v8;
  v19 = a4[3];
  *a1 = v9;
  a1[1] = v18;
  a1[3] = v19;
  return a1;
}
