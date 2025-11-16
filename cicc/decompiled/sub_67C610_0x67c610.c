// Function: sub_67C610
// Address: 0x67c610
//
__int64 __fastcall sub_67C610(_QWORD *a1)
{
  __int64 v2; // rbx
  int v3; // r13d
  __int64 v4; // rdi
  __int64 v5; // rbx
  int v6; // r13d
  __int64 v7; // rdi
  __int64 v8; // rbx
  int v9; // r13d
  __int64 v10; // rdi
  __int64 v11; // rbx
  int v12; // r13d
  __int64 v13; // rdi
  __int64 result; // rax

  v2 = a1[3];
  if ( v2 )
  {
    v3 = dword_4D03A00;
    do
    {
      v4 = v2;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 != -1 )
        sub_67C610(v4);
    }
    while ( v2 );
  }
  v5 = a1[5];
  if ( v5 )
  {
    v6 = dword_4D03A00;
    do
    {
      v7 = v5;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 != -1 )
        sub_67C610(v7);
    }
    while ( v5 );
  }
  v8 = a1[7];
  if ( v8 )
  {
    v9 = dword_4D03A00;
    do
    {
      v10 = v8;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 != -1 )
        sub_67C610(v10);
    }
    while ( v8 );
  }
  v11 = a1[9];
  if ( v11 )
  {
    v12 = dword_4D03A00;
    do
    {
      v13 = v11;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v12 != -1 )
        sub_67C610(v13);
    }
    while ( v11 );
  }
  if ( a1[23] )
  {
    *(_QWORD *)(a1[24] + 8LL) = qword_4D039F0;
    qword_4D039F0 = a1[23];
  }
  result = qword_4D039F8;
  qword_4D039F8 = (__int64)a1;
  a1[1] = result;
  return result;
}
