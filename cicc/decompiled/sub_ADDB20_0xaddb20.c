// Function: sub_ADDB20
// Address: 0xaddb20
//
__int64 __fastcall sub_ADDB20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 result; // rax
  __int64 v8; // r13
  __int64 v10; // rsi
  __int64 *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rsi

  v6 = *(__int64 **)a1;
  result = *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + 8 * result;
  if ( *(_QWORD *)a1 != v8 )
  {
    do
    {
      if ( a2 )
      {
        v10 = *v6;
        *a2 = *v6;
        if ( v10 )
        {
          sub_B976B0(v6, v10, a2, a4, a5, a6);
          *v6 = 0;
        }
      }
      ++v6;
      ++a2;
    }
    while ( (__int64 *)v8 != v6 );
    v11 = *(__int64 **)a1;
    result = *(unsigned int *)(a1 + 8);
    v12 = *(_QWORD *)a1 + 8 * result;
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v13 = *(_QWORD *)(v12 - 8);
        v12 -= 8;
        if ( v13 )
          result = sub_B91220(v12);
      }
      while ( (__int64 *)v12 != v11 );
    }
  }
  return result;
}
