// Function: sub_A04EA0
// Address: 0xa04ea0
//
unsigned __int64 __fastcall sub_A04EA0(__int64 *a1, unsigned int a2)
{
  __int64 v2; // r12
  unsigned __int64 result; // rax
  __int64 v5; // r14
  __int64 i; // rdx
  __int64 *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // rsi
  int v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *a1;
  result = *(unsigned int *)(*a1 + 8);
  if ( a2 != result )
  {
    v5 = 8LL * a2;
    if ( a2 < result )
    {
      v13 = *(_QWORD *)v2 + 8 * result;
      v14 = *(_QWORD *)v2 + v5;
      while ( v14 != v13 )
      {
        v15 = *(_QWORD *)(v13 - 8);
        v13 -= 8;
        if ( v15 )
          result = sub_B91220(v13);
      }
    }
    else
    {
      if ( a2 > (unsigned __int64)*(unsigned int *)(v2 + 12) )
      {
        v7 = (__int64 *)sub_C8D7D0(*a1, v2 + 16, a2, 8, v17);
        sub_A04E10(v2, v7, v8, v9, v10, v11);
        v12 = v17[0];
        if ( v2 + 16 != *(_QWORD *)v2 )
        {
          v16 = v17[0];
          _libc_free(*(_QWORD *)v2, v7);
          v12 = v16;
        }
        *(_DWORD *)(v2 + 12) = v12;
        result = *(unsigned int *)(v2 + 8);
        *(_QWORD *)v2 = v7;
      }
      result = *(_QWORD *)v2 + 8 * result;
      for ( i = v5 + *(_QWORD *)v2; i != result; result += 8LL )
      {
        if ( result )
          *(_QWORD *)result = 0;
      }
    }
    *(_DWORD *)(v2 + 8) = a2;
  }
  return result;
}
