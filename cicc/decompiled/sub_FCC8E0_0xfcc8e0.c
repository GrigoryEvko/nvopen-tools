// Function: sub_FCC8E0
// Address: 0xfcc8e0
//
__int64 __fastcall sub_FCC8E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 result; // rax
  __int64 v20; // rcx
  __int64 j; // r13
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 k; // r15
  __int64 v26; // rcx
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 i; // [rsp+8h] [rbp-38h]

  v8 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v9 = *(__int64 **)(a2 - 8);
    v10 = &v9[v8];
  }
  else
  {
    v9 = (__int64 *)(a2 - v8 * 8);
    v10 = (__int64 *)a2;
  }
  for ( ; v10 != v9; v9 += 4 )
  {
    if ( *v9 )
    {
      v11 = sub_FC8800(a1, *v9, a3, (__int64)a4, a5, a6);
      if ( *v9 )
      {
        a4 = (__int64 *)v9[2];
        a3 = v9[1];
        *a4 = a3;
        if ( a3 )
        {
          a4 = (__int64 *)v9[2];
          *(_QWORD *)(a3 + 16) = a4;
        }
      }
      *v9 = v11;
      if ( v11 )
      {
        a3 = *(_QWORD *)(v11 + 16);
        a4 = (__int64 *)(v11 + 16);
        v9[1] = a3;
        if ( a3 )
          *(_QWORD *)(a3 + 16) = v9 + 1;
        v9[2] = (__int64)a4;
        *(_QWORD *)(v11 + 16) = v9;
      }
    }
  }
  sub_FCBCE0(a1, a2);
  if ( *(_QWORD *)(a1 + 8) )
  {
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, a2, v12, v13);
      v16 = *(_QWORD *)(a2 + 96);
      v17 = v16 + 40LL * *(_QWORD *)(a2 + 104);
      if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      {
        sub_B2C6D0(a2, a2, v12, v26);
        v16 = *(_QWORD *)(a2 + 96);
      }
    }
    else
    {
      v16 = *(_QWORD *)(a2 + 96);
      v17 = v16 + 40LL * *(_QWORD *)(a2 + 104);
    }
    for ( ;
          v17 != v16;
          *(_QWORD *)(v16 - 32) = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
                                    *(_QWORD *)(a1 + 8),
                                    v18) )
    {
      v18 = *(_QWORD *)(v16 + 8);
      v16 += 40;
    }
  }
  result = *(_QWORD *)(a2 + 80);
  v20 = a2 + 72;
  v27 = a2 + 72;
  for ( i = result; v27 != result; i = result )
  {
    if ( !i )
      BUG();
    for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
      {
        sub_FCBDE0(a1, 0, v12, v20, v14, v15);
        BUG();
      }
      sub_FCBDE0(a1, (unsigned __int8 *)(j - 24), v12, v20, v14, v15);
      v22 = *(_QWORD *)(j + 40);
      if ( v22 )
      {
        v23 = sub_B14240(v22);
        v24 = v12;
        for ( k = v23; v24 != k; k = *(_QWORD *)(k + 8) )
          sub_FCC310((_DWORD *)a1, k);
      }
    }
    result = *(_QWORD *)(i + 8);
  }
  return result;
}
