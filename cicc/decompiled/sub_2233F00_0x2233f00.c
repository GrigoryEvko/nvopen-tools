// Function: sub_2233F00
// Address: 0x2233f00
//
__int64 __fastcall sub_2233F00(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // rdi
  unsigned __int8 *v4; // rax

  result = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD **)a1;
  if ( (_DWORD)result == -1 && v3 )
  {
    v4 = (unsigned __int8 *)v3[2];
    if ( (unsigned __int64)v4 >= v3[3] )
    {
      result = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 72LL))(v3);
      if ( (_DWORD)result == -1 )
        *(_QWORD *)a1 = 0;
    }
    else
    {
      return *v4;
    }
  }
  return result;
}
