// Function: sub_9C3E90
// Address: 0x9c3e90
//
__int64 __fastcall sub_9C3E90(__int64 a1, unsigned __int64 a2, unsigned int a3, _QWORD *a4)
{
  __int64 v5; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r13

  if ( a3 > a2 )
    return 1;
  v5 = 8LL * a3;
  v7 = 8 * a2 - v5;
  v8 = a4[1];
  v9 = v5 + a1;
  v10 = v7 >> 3;
  if ( (unsigned __int64)((v7 >> 3) + v8) > a4[2] )
  {
    sub_C8D290(a4, a4 + 3, v10 + v8, 1);
    v8 = a4[1];
  }
  v11 = v8 + *a4;
  if ( v7 > 0 )
  {
    v12 = 0;
    do
    {
      *(_BYTE *)(v11 + v12) = *(_QWORD *)(v9 + 8 * v12);
      ++v12;
    }
    while ( v10 - v12 > 0 );
    v8 = a4[1];
  }
  v13 = v8 + v10;
  a4[1] = v13;
  return 0;
}
