// Function: sub_13226C0
// Address: 0x13226c0
//
__int64 __fastcall sub_13226C0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v6; // rsi
  _QWORD *v7; // rdi
  _QWORD *v8; // rbx
  int *v9; // r14
  unsigned int v10; // eax
  int v11; // edx
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  _QWORD *v16; // rdi
  __int64 result; // rax
  _QWORD *v18; // [rsp+10h] [rbp-40h]

  v6 = *(_QWORD **)(a1 + 80);
  if ( a3 )
  {
    v18 = *(_QWORD **)(a2 + 80);
  }
  else
  {
    *(_DWORD *)(a1 + 24) += *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 56) += *(_QWORD *)(a2 + 56);
    *(_QWORD *)(a1 + 64) += *(_QWORD *)(a2 + 64);
    *(_QWORD *)(a1 + 72) += *(_QWORD *)(a2 + 72);
    v7 = *(_QWORD **)(a2 + 80);
    v6[3] += v7[3];
    v6[18] += v7[18];
    v6[11] += v7[11];
    v18 = v7;
  }
  v6[12] += v18[12];
  v6[13] += v18[13];
  v6[14] += v18[14];
  v6[15] += v18[15];
  v6[16] += v18[16];
  v6[17] += v18[17];
  sub_131DF20((__int64)(v6 + 23), (__int64)(v18 + 23));
  sub_131DF20((__int64)(v6 + 31), (__int64)(v18 + 31));
  sub_131DF20((__int64)(v6 + 39), (__int64)(v18 + 39));
  sub_131DF20((__int64)(v6 + 47), (__int64)(v18 + 47));
  sub_131DF20((__int64)(v6 + 55), (__int64)(v18 + 55));
  sub_131DF20((__int64)(v6 + 63), (__int64)(v18 + 63));
  sub_131DF20((__int64)(v6 + 71), (__int64)(v18 + 71));
  sub_131DF20((__int64)(v6 + 79), (__int64)(v18 + 79));
  sub_131DF20((__int64)(v6 + 87), (__int64)(v18 + 87));
  sub_131DF20((__int64)(v6 + 95), (__int64)(v18 + 95));
  sub_131DF20((__int64)(v6 + 103), (__int64)(v18 + 103));
  sub_131DF20((__int64)(v6 + 111), (__int64)(v18 + 111));
  if ( !a3 )
  {
    *v6 += *v18;
    v6[1] += v18[1];
    v6[2] += v18[2];
    v6[4] += v18[4];
    v6[1296] += v18[1296];
  }
  v6[1297] += v18[1297];
  v6[1298] += v18[1298];
  v6[1299] += v18[1299];
  v6[1300] += v18[1300];
  v6[1301] += v18[1301];
  if ( !a3 )
    v6[5] += v18[5];
  v6[6] += v18[6];
  v6[7] += v18[7];
  v6[10] += v18[10];
  v6[9] += v18[9];
  v6[20] += v18[20];
  v6[21] += v18[21];
  v6[22] += v18[22];
  if ( !*(_DWORD *)a2 )
    v6[1295] = v18[1295];
  v8 = (_QWORD *)((char *)v6 + 10532);
  v9 = (int *)v18 + 2633;
  do
  {
    *(_QWORD *)((char *)v8 - 116) += *(_QWORD *)(v9 - 29);
    *(_QWORD *)((char *)v8 - 108) += *(_QWORD *)(v9 - 27);
    *(_QWORD *)((char *)v8 - 100) += *(_QWORD *)(v9 - 25);
    if ( !a3 )
      *(_QWORD *)((char *)v8 - 92) += *(_QWORD *)(v9 - 23);
    *(_QWORD *)((char *)v8 - 84) += *(_QWORD *)(v9 - 21);
    *(_QWORD *)((char *)v8 - 76) += *(_QWORD *)(v9 - 19);
    *(_QWORD *)((char *)v8 - 68) += *(_QWORD *)(v9 - 17);
    *(_QWORD *)((char *)v8 - 60) += *(_QWORD *)(v9 - 15);
    if ( !a3 )
    {
      *(_QWORD *)((char *)v8 - 52) += *(_QWORD *)(v9 - 13);
      *(_QWORD *)((char *)v8 - 44) += *(_QWORD *)(v9 - 11);
    }
    sub_130B1D0((_QWORD *)((char *)v8 - 36), (__int64 *)(v9 - 9));
    if ( (int)sub_130B150((_QWORD *)((char *)v8 - 28), v9 - 7) < 0 )
      sub_130B140((_QWORD *)((char *)v8 - 28), (__int64 *)(v9 - 7));
    *(_QWORD *)((char *)v8 - 20) += *(_QWORD *)(v9 - 5);
    *(_QWORD *)((char *)v8 - 12) += *(_QWORD *)(v9 - 3);
    v10 = *(v9 - 1);
    if ( *((_DWORD *)v8 - 1) < v10 )
      *((_DWORD *)v8 - 1) = v10;
    v11 = *v9;
    v9 += 36;
    *(_DWORD *)v8 += v11;
    *(_QWORD *)((char *)v8 + 4) += *(_QWORD *)(v9 - 35);
    *(_QWORD *)((char *)v8 + 20) += *(_QWORD *)(v9 - 31);
    v8 += 18;
  }
  while ( v8 != (_QWORD *)((char *)v6 + 15716) );
  v12 = v18 + 1950;
  v13 = v6 + 1950;
  v14 = v6 + 3126;
  do
  {
    *v13 += *v12;
    v13[1] += v12[1];
    v13[2] += v12[2];
    if ( !a3 )
      v13[5] += v12[5];
    v13 += 6;
    v12 += 6;
  }
  while ( v14 != v13 );
  v15 = v18 + 3126;
  v16 = v6 + 4320;
  do
  {
    *v14 += *v15;
    v15 += 6;
    v14[2] += *(v15 - 4);
    v14[4] += *(v15 - 2);
    v14[1] += *(v15 - 5);
    v14[3] += *(v15 - 3);
    v14[5] += *(v15 - 1);
    v14 += 6;
  }
  while ( v16 != v14 );
  sub_1348600(v16, v18 + 4320);
  result = v18[4720];
  v6[4720] += result;
  return result;
}
