// Function: sub_C89030
// Address: 0xc89030
//
__int64 __fastcall sub_C89030(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  unsigned int v3; // r12d
  __int64 v4; // r14
  __int64 v5; // r13

  result = 1;
  v3 = *((_DWORD *)a1 + 2);
  if ( v3 )
  {
    v4 = *a1;
    v5 = sub_CBADF0(v3, *a1, 0, 0);
    sub_22410F0(a2, v5 - 1, 0);
    sub_CBADF0(v3, v4, *a2, v5);
    return 0;
  }
  return result;
}
