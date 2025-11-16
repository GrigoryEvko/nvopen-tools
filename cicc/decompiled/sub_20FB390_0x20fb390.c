// Function: sub_20FB390
// Address: 0x20fb390
//
__int64 __fastcall sub_20FB390(_QWORD *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 i; // r12
  unsigned __int8 *v5; // rax
  __int64 v6; // rax

  if ( a3 )
  {
    for ( i = a3; ; i = *(_QWORD *)(i - 8) )
    {
      v5 = sub_15B1000(a2);
      if ( *(_DWORD *)(*(_QWORD *)&v5[8 * (5LL - *((unsigned int *)v5 + 2))] + 36LL) )
      {
        sub_20FAC60((__int64)a1, (__int64)a2);
        return sub_20FAF80(a1, (__int64)a2, i);
      }
      v6 = *(unsigned int *)(i + 8);
      if ( (_DWORD)v6 != 2 )
        break;
      a2 = *(unsigned __int8 **)(i - 16);
      if ( !*(_QWORD *)(i - 8) )
        return sub_20FB440(a1, a2);
    }
    a2 = *(unsigned __int8 **)(i - 8 * v6);
  }
  return sub_20FB440(a1, a2);
}
