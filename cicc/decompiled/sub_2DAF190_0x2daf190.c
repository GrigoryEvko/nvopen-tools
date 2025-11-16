// Function: sub_2DAF190
// Address: 0x2daf190
//
__int64 __fastcall sub_2DAF190(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = v6 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  result = 5LL * (unsigned int)sub_2E88FE0(a2);
  v9 = v6 + 8 * result;
  if ( v7 != v9 )
  {
    v10 = a2;
    do
    {
      if ( !*(_BYTE *)v9 )
      {
        result = *(unsigned int *)(v9 + 8);
        if ( (int)result < 0 )
        {
          v15 = v10;
          v11 = sub_2DAE370(a1, v10, a3, a4, v9);
          result = sub_2DAEF70((__int64)a1, v9, v11, v12, v13, v14);
          v10 = v15;
        }
      }
      v9 += 40;
    }
    while ( v7 != v9 );
  }
  return result;
}
