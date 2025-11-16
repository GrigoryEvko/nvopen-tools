// Function: sub_27E4FC0
// Address: 0x27e4fc0
//
__int64 __fastcall sub_27E4FC0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rdx

  if ( *(_BYTE *)(a1 + 124) )
  {
    v4 = *(_QWORD **)(a1 + 104);
    v5 = &v4[*(unsigned int *)(a1 + 116)];
    if ( v4 == v5 )
      return sub_27E44B0(a1, a2, a3);
    while ( a2 != *v4 )
    {
      if ( v5 == ++v4 )
        return sub_27E44B0(a1, a2, a3);
    }
    return 0;
  }
  if ( sub_C8CA60(a1 + 96, a2) )
    return 0;
  return sub_27E44B0(a1, a2, a3);
}
