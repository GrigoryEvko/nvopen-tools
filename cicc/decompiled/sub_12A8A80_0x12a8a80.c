// Function: sub_12A8A80
// Address: 0x12a8a80
//
unsigned __int64 __fastcall sub_12A8A80(_QWORD *a1, int a2, _BYTE *a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned int i; // r12d
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v20; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v21; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v24[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  result = (unsigned __int64)&v23;
  if ( a2 )
  {
    for ( i = 0; i != a2; ++i )
    {
      v7 = (__int64 *)(*a1 + 48LL);
      v25 = 257;
      v8 = sub_12A8800(v7, a4, a3, i, (__int64)v24);
      v9 = a1[1];
      v10 = (_QWORD *)*a1;
      v20 = v8;
      v25 = 257;
      v11 = sub_1648A60(64, 1);
      v12 = (_QWORD *)v11;
      if ( v11 )
        sub_15F9210(v11, a4, v20, 0, 0, 0);
      v13 = v10[7];
      if ( v13 )
      {
        v21 = (unsigned __int64 *)v10[8];
        sub_157E9D0(v13 + 40, v12);
        v14 = *v21;
        v15 = v12[3] & 7LL;
        v12[4] = v21;
        v14 &= 0xFFFFFFFFFFFFFFF8LL;
        v12[3] = v14 | v15;
        *(_QWORD *)(v14 + 8) = v12 + 3;
        *v21 = *v21 & 7 | (unsigned __int64)(v12 + 3);
      }
      sub_164B780(v12, v24);
      v16 = v10[6];
      if ( v16 )
      {
        v23 = v10[6];
        sub_1623A60(&v23, v16, 2);
        if ( v12[6] )
          sub_161E7C0(v12 + 6);
        v17 = v23;
        v12[6] = v23;
        if ( v17 )
          sub_1623210(&v23, v17, v12 + 6);
      }
      result = *(unsigned int *)(v9 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(v9 + 12) )
      {
        sub_16CD150(v9, v9 + 16, 0, 8);
        result = *(unsigned int *)(v9 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v9 + 8 * result) = v12;
      ++*(_DWORD *)(v9 + 8);
    }
  }
  return result;
}
