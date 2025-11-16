// Function: sub_13C9060
// Address: 0x13c9060
//
__int64 __fastcall sub_13C9060(__int64 a1, __int64 a2)
{
  __int16 v2; // dx
  __int64 *v3; // rax
  __int64 result; // rax
  _QWORD *v5; // rbx
  _QWORD *i; // r12

  v2 = *(_WORD *)(a1 + 24);
  if ( v2 == 7 )
  {
    while ( *(_QWORD *)(a1 + 48) != a2 )
    {
      v3 = *(__int64 **)(a1 + 32);
      a1 = *v3;
      v2 = *(_WORD *)(*v3 + 24);
      if ( v2 != 7 )
        goto LABEL_4;
    }
    return a1;
  }
  else
  {
LABEL_4:
    result = 0;
    if ( v2 == 4 )
    {
      v5 = *(_QWORD **)(a1 + 32);
      for ( i = &v5[*(_QWORD *)(a1 + 40)]; i != v5; ++v5 )
      {
        result = sub_13C9060(*v5, a2);
        if ( result )
          break;
      }
    }
  }
  return result;
}
