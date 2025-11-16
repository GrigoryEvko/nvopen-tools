// Function: sub_D69420
// Address: 0xd69420
//
__int64 __fastcall sub_D69420(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 result; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rax

  v2 = *a1;
  v3 = *(_QWORD *)(a2 + 64);
  result = sub_D68C20(*a1, v3);
  if ( result )
  {
    if ( *(_BYTE *)a2 != 26 )
    {
      v5 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 && result != v5 )
        return v5 - 48;
      return 0;
    }
    v6 = sub_D68BB0(v2, v3);
    v7 = (_QWORD *)(*(_QWORD *)(a2 + 32) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (_QWORD *)v6 == v7 )
      return 0;
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      if ( *((_BYTE *)v7 - 32) != 26 )
        break;
      v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)v6 == v7 )
        return 0;
    }
    return (__int64)(v7 - 4);
  }
  return result;
}
