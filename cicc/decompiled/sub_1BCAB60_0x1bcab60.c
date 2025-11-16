// Function: sub_1BCAB60
// Address: 0x1bcab60
//
unsigned __int64 __fastcall sub_1BCAB60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // rdx
  unsigned __int64 result; // rax

  v6 = (_DWORD *)a1[1];
  result = (unsigned int)*v6;
  *v6 = result + 1;
  *(_DWORD *)(*(_QWORD *)(a2 + 8) + 84LL) = result;
  if ( a2 == *(_QWORD *)(a2 + 8) )
  {
    sub_1BC9CB0(a1[3], a2, 0, *a1, a5, a6);
    result = a1[2];
    ++*(_DWORD *)result;
  }
  return result;
}
