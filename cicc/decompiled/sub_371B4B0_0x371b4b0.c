// Function: sub_371B4B0
// Address: 0x371b4b0
//
__int64 __fastcall sub_371B4B0(__int64 a1, __int64 a2)
{
  _QWORD *i; // r13
  __int64 v4; // rdi
  __int64 *v5; // r12
  __int64 result; // rax
  __int64 *v7; // rbx

  for ( i = (_QWORD *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        (_QWORD *)(a2 + 48) != i;
        i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v4 = *(_QWORD *)(a1 + 24);
    if ( !i )
    {
      sub_3188190(v4, 0);
      BUG();
    }
    v5 = i - 3;
    result = sub_3188190(v4, (__int64)(i - 3));
    if ( (*((_BYTE *)i - 17) & 0x40) != 0 )
    {
      v7 = (__int64 *)*(i - 4);
      v5 = &v7[4 * (*((_DWORD *)i - 5) & 0x7FFFFFF)];
    }
    else
    {
      result = 32LL * (*((_DWORD *)i - 5) & 0x7FFFFFF);
      v7 = (__int64 *)((char *)v5 - result);
    }
    for ( ; v5 != v7; v7 += 4 )
    {
      if ( *(_BYTE *)*v7 != 23 )
        result = sub_3188190(*(_QWORD *)(a1 + 24), *v7);
    }
  }
  return result;
}
