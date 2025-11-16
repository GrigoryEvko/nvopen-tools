// Function: sub_2E01300
// Address: 0x2e01300
//
__int64 __fastcall sub_2E01300(__int64 a1, int a2, unsigned int *a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v5; // r13d
  __int64 v6; // r15
  int v7; // ebx
  unsigned __int64 v8; // r9
  char v9; // r14
  __int64 v10; // r13
  unsigned int v11; // ebx
  __int64 v12; // r12
  int *v13; // r12
  __int64 v14; // rbx
  int *v15; // r13
  int v16; // esi
  char v19; // [rsp+1Bh] [rbp-35h]
  unsigned int v20; // [rsp+1Ch] [rbp-34h]

  sub_2E00C30(a1, a2, a3, a4);
  result = sub_2DF47B0(a1, a2);
  v19 = 0;
  if ( result )
  {
    v5 = a2;
    v6 = result;
    do
    {
      result = a1;
      v7 = *(_DWORD *)(v6 + 64);
      if ( v7 )
      {
        v8 = v5;
        v9 = 0;
        v10 = *(_QWORD *)(a1 + 112);
        v11 = v7 - 1;
        v12 = 40LL * v11;
        while ( 1 )
        {
          result = v12 + *(_QWORD *)(v6 + 56);
          if ( !*(_BYTE *)result && *(_DWORD *)(result + 8) == (_DWORD)v8 )
          {
            v20 = v8;
            result = sub_2DFEAA0(v6, v11, (unsigned __int64)a3, a4, v10, v8);
            v8 = v20;
            v9 |= result;
          }
          v12 -= 40;
          if ( !v11 )
            break;
          --v11;
        }
        v19 |= v9;
        v5 = v8;
      }
      v6 = *(_QWORD *)(v6 + 48);
    }
    while ( v6 );
    if ( v19 )
    {
      v13 = (int *)a3;
      v14 = sub_2DF47B0(a1, v5);
      result = a4;
      v15 = (int *)&a3[a4];
      if ( a3 != (unsigned int *)v15 )
      {
        do
        {
          v16 = *v13++;
          result = sub_2DF7150(a1, v16, v14);
        }
        while ( v15 != v13 );
      }
    }
  }
  return result;
}
