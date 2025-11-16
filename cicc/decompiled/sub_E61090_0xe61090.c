// Function: sub_E61090
// Address: 0xe61090
//
__int64 __fastcall sub_E61090(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  _DWORD *v6; // rdx
  __int64 result; // rax
  unsigned __int64 v8; // rsi
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rdi

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 288);
  v4 = *(_QWORD *)(a1 + 280);
  v5 = 0x6DB6DB6DB6DB6DB7LL * ((v3 - v4) >> 3);
  if ( a2 >= v5 )
  {
    v8 = a2 + 1;
    if ( v8 > v5 )
    {
      sub_E60D70((__int64 *)(a1 + 280), v8 - v5);
      v4 = *(_QWORD *)(a1 + 280);
    }
    else if ( v8 < v5 )
    {
      v10 = v4 + 56 * v8;
      if ( v3 != v10 )
      {
        v11 = v4 + 56 * v8;
        do
        {
          v12 = *(unsigned int *)(v11 + 48);
          v13 = *(_QWORD *)(v11 + 32);
          v11 += 56;
          sub_C7D6A0(v13, 16 * v12, 4);
        }
        while ( v3 != v11 );
        *(_QWORD *)(a1 + 288) = v10;
        v4 = *(_QWORD *)(a1 + 280);
      }
    }
  }
  v6 = (_DWORD *)(v4 + 56 * v2);
  result = 0;
  if ( !*v6 )
  {
    *v6 = -1;
    return 1;
  }
  return result;
}
