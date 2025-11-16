// Function: sub_16A5130
// Address: 0x16a5130
//
unsigned __int64 __fastcall sub_16A5130(unsigned __int64 *a1, unsigned int a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rdi

  v3 = ((unsigned __int64)a2 + 63) >> 6;
  v4 = *((unsigned int *)a1 + 2);
  result = (unsigned __int64)(v4 + 63) >> 6;
  if ( (_DWORD)v3 == (_DWORD)result )
  {
    *((_DWORD *)a1 + 2) = a2;
  }
  else
  {
    if ( (unsigned int)v4 > 0x40 )
    {
      v6 = *a1;
      if ( v6 )
        result = j_j___libc_free_0_0(v6);
    }
    *((_DWORD *)a1 + 2) = a2;
    if ( a2 > 0x40 )
    {
      result = sub_2207820(8 * v3);
      *a1 = result;
    }
  }
  return result;
}
