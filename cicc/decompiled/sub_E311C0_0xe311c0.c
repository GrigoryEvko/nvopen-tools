// Function: sub_E311C0
// Address: 0xe311c0
//
__int64 __fastcall sub_E311C0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r14d
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  char *v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rcx

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(result + 24) )
  {
    v4 = a3;
    sub_E311A0(a1, (__int64)a2, a3, **(_QWORD ***)(result + 16));
    result = *(_QWORD *)(a1 + 16);
    if ( *(_QWORD *)(result + 24) > 1u )
    {
      v7 = 1;
      do
      {
        v8 = a2[1];
        v9 = a2[2];
        v10 = (char *)*a2;
        if ( v8 + 2 > v9 )
        {
          v11 = v8 + 994;
          v12 = 2 * v9;
          if ( v11 > v12 )
            a2[2] = v11;
          else
            a2[2] = v12;
          v13 = realloc(v10);
          *a2 = v13;
          v10 = (char *)v13;
          if ( !v13 )
            abort();
          v8 = a2[1];
        }
        *(_WORD *)&v10[v8] = 23389;
        a2[1] += 2;
        v14 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) + 8 * v7++);
        sub_E311A0(a1, (__int64)a2, v4, v14);
        result = *(_QWORD *)(a1 + 16);
      }
      while ( *(_QWORD *)(result + 24) > v7 );
    }
  }
  return result;
}
