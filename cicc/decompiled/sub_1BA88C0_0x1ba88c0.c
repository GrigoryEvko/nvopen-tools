// Function: sub_1BA88C0
// Address: 0x1ba88c0
//
unsigned __int64 __fastcall sub_1BA88C0(__int64 a1, unsigned int a2, double a3)
{
  float v3; // xmm3_4
  int v4; // eax
  double v5; // xmm1_8
  unsigned int v6; // r14d
  int v7; // ebx
  bool v8; // r13
  unsigned __int64 v9; // rax
  bool v10; // cc
  __int64 v11; // rax
  __int64 v13; // rbx
  _QWORD *v14; // r13
  __int64 v15; // rax
  float v16; // [rsp+8h] [rbp-218h]
  _QWORD v17[11]; // [rsp+10h] [rbp-210h] BYREF
  _BYTE v18[440]; // [rsp+68h] [rbp-1B8h] BYREF

  v3 = (float)(int)sub_1BA8260(a1, 1u);
  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 376) + 40LL);
  v16 = v3;
  if ( a2 <= 1 )
  {
    *(_QWORD *)&v5 = LODWORD(v3);
    v7 = 1;
  }
  else
  {
    LODWORD(v5) = 2139095039;
    if ( v4 != 1 )
      *(float *)&v5 = v3;
    v6 = 2;
    v7 = 1;
    v8 = v4 != 1;
    do
    {
      v9 = sub_1BA8260(a1, v6);
      HIDWORD(a3) = 0;
      *(_QWORD *)&v5 = LODWORD(v5);
      *(float *)&a3 = (float)(int)v9 / (float)(int)v6;
      if ( BYTE4(v9) == 1 || !v8 )
      {
        v10 = *(float *)&v5 <= *(float *)&a3;
        *(float *)&a3 = fminf(*(float *)&a3, *(float *)&v5);
        if ( !v10 )
          v7 = v6;
        v5 = a3;
      }
      v6 *= 2;
    }
    while ( a2 >= v6 );
  }
  if ( !byte_4FB8200 && *(_DWORD *)a1 )
  {
    v13 = *(_QWORD *)(a1 + 296);
    v14 = *(_QWORD **)(a1 + 360);
    v15 = sub_1BF18B0(*(_QWORD *)(a1 + 376), a3, v5);
    sub_1BF1750(v17, v15, "ConditionalStore", 16, v13, 0);
    sub_15CAB20((__int64)v17, "store that is conditionally executed prevents vectorization", 0x3Bu);
    sub_143AA50(v14, (__int64)v17);
    v17[0] = &unk_49ECF68;
    sub_1897B80((__int64)v18);
    v11 = 1;
  }
  else
  {
    v11 = (unsigned int)v7;
    v16 = (float)v7 * *(float *)&v5;
  }
  return ((unsigned __int64)(unsigned int)(int)v16 << 32) | v11;
}
