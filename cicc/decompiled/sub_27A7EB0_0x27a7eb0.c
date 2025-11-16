// Function: sub_27A7EB0
// Address: 0x27a7eb0
//
__int64 __fastcall sub_27A7EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __int64 v7; // r13
  __int64 i; // r15
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v20; // [rsp+20h] [rbp-70h]
  __int64 v21; // [rsp+28h] [rbp-68h]
  __m128i v22; // [rsp+40h] [rbp-50h] BYREF
  __int64 v23; // [rsp+50h] [rbp-40h] BYREF
  __int64 v24; // [rsp+58h] [rbp-38h]

  v7 = a1;
  v21 = (a3 - 1) / 2;
  v20 = a3 & 1;
  if ( a2 >= v21 )
  {
    v10 = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
    {
      v23 = a4;
      v24 = a5;
      goto LABEL_13;
    }
    v12 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = a1 + 32 * (i + 1);
    if ( (unsigned __int8)sub_27A2220((__int64 *)&a7, (int *)v10, a1 + 16 * (v9 - 1)) )
      v10 = a1 + 16 * --v9;
    v11 = a1 + 16 * i;
    *(_DWORD *)v11 = *(_DWORD *)v10;
    *(_QWORD *)(v11 + 8) = *(_QWORD *)(v10 + 8);
    if ( v9 >= v21 )
      break;
  }
  v12 = v9;
  v7 = a1;
  if ( !v20 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v12 )
    {
      v12 = 2 * v12 + 1;
      v16 = v7 + 16 * v12;
      *(_DWORD *)v10 = *(_DWORD *)v16;
      *(_QWORD *)(v10 + 8) = *(_QWORD *)(v16 + 8);
      v10 = v16;
    }
  }
  v23 = a4;
  v22 = _mm_loadu_si128((const __m128i *)&a7);
  v24 = a5;
  v13 = (v12 - 1) / 2;
  if ( v12 > a2 )
  {
    while ( 1 )
    {
      v10 = v7 + 16 * v12;
      v14 = v7 + 16 * v13;
      if ( !(unsigned __int8)sub_27A2220(v22.m128i_i64, (int *)v14, (__int64)&v23) )
        break;
      v12 = v13;
      *(_DWORD *)v10 = *(_DWORD *)v14;
      *(_QWORD *)(v10 + 8) = *(_QWORD *)(v14 + 8);
      if ( a2 >= v13 )
      {
        v10 = v7 + 16 * v13;
        break;
      }
      v13 = (v13 - 1) / 2;
    }
  }
LABEL_13:
  *(_DWORD *)v10 = v23;
  result = v24;
  *(_QWORD *)(v10 + 8) = v24;
  return result;
}
