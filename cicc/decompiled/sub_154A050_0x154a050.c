// Function: sub_154A050
// Address: 0x154a050
//
__int64 __fastcall sub_154A050(unsigned int a1, __int64 a2)
{
  __int64 result; // rax
  void *v3; // rdx

  if ( a1 == 3 )
    return sub_1263B40(a2, "thread_local(initialexec) ");
  if ( a1 > 3 )
  {
    if ( a1 == 4 )
      return sub_1263B40(a2, "thread_local(localexec) ");
  }
  else if ( a1 == 1 )
  {
    v3 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0xCu )
    {
      return sub_16E7EE0(a2, "thread_local ", 13);
    }
    else
    {
      qmemcpy(v3, "thread_local ", 13);
      *(_QWORD *)(a2 + 24) += 13LL;
      return 0x6C5F646165726874LL;
    }
  }
  else if ( a1 == 2 )
  {
    return sub_1263B40(a2, "thread_local(localdynamic) ");
  }
  return result;
}
