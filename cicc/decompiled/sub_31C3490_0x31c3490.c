// Function: sub_31C3490
// Address: 0x31c3490
//
__int64 __fastcall sub_31C3490(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi

  result = *(_QWORD *)a1;
  v3 = *(unsigned int *)(a1 + 8);
  if ( v3 * 8 )
  {
    v4 = &a2[v3];
    do
    {
      if ( a2 )
      {
        *a2 = *(_QWORD *)result;
        *(_QWORD *)result = 0;
      }
      ++a2;
      result += 8;
    }
    while ( v4 != a2 );
    v5 = *(_QWORD *)a1;
    result = *(unsigned int *)(a1 + 8);
    v6 = *(_QWORD *)a1 + 8 * result;
    while ( v5 != v6 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 - 8);
        v6 -= 8;
        if ( !v7 )
          break;
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
        if ( v5 == v6 )
          return result;
      }
    }
  }
  return result;
}
