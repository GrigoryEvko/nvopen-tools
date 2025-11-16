// Function: sub_3911C30
// Address: 0x3911c30
//
__int64 __fastcall sub_3911C30(__int64 a1, unsigned int a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  _DWORD *v6; // rdx
  __int64 result; // rax
  unsigned __int64 v8; // rsi
  __int64 v10; // r14
  __int64 v11; // r15
  unsigned __int64 v12; // rdi

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 296);
  v4 = *(_QWORD *)(a1 + 288);
  v5 = 0x6DB6DB6DB6DB6DB7LL * ((v3 - v4) >> 3);
  if ( a2 >= v5 )
  {
    v8 = a2 + 1;
    if ( v8 > v5 )
    {
      sub_3911920((unsigned __int64 *)(a1 + 288), v8 - v5);
      v4 = *(_QWORD *)(a1 + 288);
    }
    else if ( v8 < v5 )
    {
      v10 = v4 + 56 * v8;
      if ( v3 != v10 )
      {
        v11 = v4 + 56 * v8;
        do
        {
          v12 = *(_QWORD *)(v11 + 32);
          v11 += 56;
          j___libc_free_0(v12);
        }
        while ( v3 != v11 );
        *(_QWORD *)(a1 + 296) = v10;
        v4 = *(_QWORD *)(a1 + 288);
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
