// Function: sub_28ED3E0
// Address: 0x28ed3e0
//
__int64 __fastcall sub_28ED3E0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  unsigned __int8 *i; // rbx
  __int64 result; // rax
  __int64 v9; // rdx

  v6 = *a1;
  for ( i = a1; (unsigned __int8)v6 > 0x1Cu; v6 = *i )
  {
    if ( (unsigned int)(v6 - 42) > 0x11 )
      break;
    v9 = *((_QWORD *)i + 2);
    if ( !v9
      || *(_QWORD *)(v9 + 8)
      || (unsigned int)(v6 - 46) > 1
      || (unsigned __int8)sub_920620((__int64)i) && (!sub_B451B0((__int64)i) || !sub_B451E0((__int64)i)) )
    {
      break;
    }
    sub_28ED3E0(*((_QWORD *)i - 4), a2);
    i = (unsigned __int8 *)*((_QWORD *)i - 8);
  }
  result = *(unsigned int *)(a2 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a2 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = i;
  ++*(_DWORD *)(a2 + 8);
  return result;
}
