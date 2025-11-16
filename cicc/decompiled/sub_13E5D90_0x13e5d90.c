// Function: sub_13E5D90
// Address: 0x13e5d90
//
__int64 __fastcall sub_13E5D90(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rax
  _BOOL4 v6; // ebx
  __int64 v7; // rax
  unsigned __int64 *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // r15d
  unsigned int i; // r12d
  __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __m128i *v17; // rsi
  __m128i v19; // [rsp+0h] [rbp-40h] BYREF

  v2 = a1 + 4;
  v3 = (_QWORD *)a1[5];
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = v3[4];
      v5 = (_QWORD *)v3[3];
      if ( a2 < v4 )
        v5 = (_QWORD *)v3[2];
      if ( !v5 )
        break;
      v3 = v5;
    }
    if ( a2 >= v4 )
    {
      if ( v4 >= a2 )
        return 0;
LABEL_9:
      v6 = 1;
      if ( v2 != v3 )
        v6 = a2 < v3[4];
      goto LABEL_11;
    }
    if ( (_QWORD *)a1[6] == v3 )
      goto LABEL_9;
LABEL_23:
    if ( a2 <= *(_QWORD *)(sub_220EF80(v3) + 32) )
      return 0;
    goto LABEL_9;
  }
  v3 = a1 + 4;
  if ( v2 != (_QWORD *)a1[6] )
    goto LABEL_23;
  v6 = 1;
LABEL_11:
  v7 = sub_22077B0(40);
  *(_QWORD *)(v7 + 32) = a2;
  sub_220F040(v6, v7, v3, v2);
  ++a1[8];
  v8 = (unsigned __int64 *)sub_22077B0(80);
  v9 = (__int64)v8;
  if ( v8 )
  {
    *v8 = a2;
    v8[1] = 0;
    v8[2] = 0;
    v8[3] = 0;
    v8[4] = 0;
    v8[5] = 0;
    v8[6] = 0;
    v8[7] = 0;
    v8[8] = 0;
    v8[9] = 0;
    v19.m128i_i64[0] = a2;
    sub_1292090((__int64)(v8 + 1), 0, &v19);
  }
  v10 = sub_157EBA0(a2);
  v11 = v10;
  if ( v10 )
  {
    v12 = sub_15F4D60(v10);
    if ( v12 )
    {
      for ( i = 0; i != v12; ++i )
      {
        v14 = i;
        v15 = sub_15F4DF0(v11, v14);
        sub_13E5870(a1, (char **)v9, v15);
      }
    }
  }
  v16 = *(_QWORD *)(v9 + 32);
  v17 = (__m128i *)a1[1];
  v19.m128i_i64[0] = v9;
  v19.m128i_i64[1] = v16;
  if ( v17 == (__m128i *)a1[2] )
  {
    sub_13E5C10((const __m128i **)a1, v17, &v19);
  }
  else
  {
    if ( v17 )
    {
      *v17 = _mm_loadu_si128(&v19);
      v17 = (__m128i *)a1[1];
    }
    a1[1] = v17 + 1;
  }
  return 1;
}
