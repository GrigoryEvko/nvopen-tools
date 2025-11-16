// Function: sub_2247910
// Address: 0x2247910
//
__int64 __fastcall sub_2247910(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // rdi
  unsigned int *v4; // rax

  result = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD **)a1;
  if ( (_DWORD)result == -1 && v3 )
  {
    v4 = (unsigned int *)v3[2];
    if ( (unsigned __int64)v4 >= v3[3] )
      result = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 72LL))(v3);
    else
      result = *v4;
    if ( (_DWORD)result == -1 )
      *(_QWORD *)a1 = 0;
  }
  return result;
}
