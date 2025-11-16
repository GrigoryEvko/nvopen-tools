// Function: sub_35033F0
// Address: 0x35033f0
//
void __fastcall sub_35033F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  int v5; // r13d
  unsigned __int64 v6; // r8
  __int64 v7; // rax
  _BYTE *v8; // rsi
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(unsigned int *)(a2 + 128);
  v3 = *(_QWORD **)(a2 + 120);
  v9[0] = a2;
  v4 = &v3[2 * v2];
  if ( v4 == v3 )
  {
    v7 = a2;
    v5 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v6 = sub_35031F0((__int64)a1, *v3 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v9[0];
      v3 += 2;
      v5 += v9[0] == v6;
    }
    while ( v4 != v3 );
  }
  *(_DWORD *)(a1[3] + 4LL * *(unsigned int *)(v7 + 200)) = v5;
  v8 = (_BYTE *)a1[7];
  if ( v8 == (_BYTE *)a1[8] )
  {
    sub_2ECAD30((__int64)(a1 + 6), v8, v9);
  }
  else
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = v9[0];
      v8 = (_BYTE *)a1[7];
    }
    a1[7] = v8 + 8;
  }
}
