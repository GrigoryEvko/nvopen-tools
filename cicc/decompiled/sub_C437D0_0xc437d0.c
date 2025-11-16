// Function: sub_C437D0
// Address: 0xc437d0
//
unsigned __int64 __fastcall sub_C437D0(unsigned __int64 *a1, _QWORD *a2, unsigned int a3)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // rcx
  unsigned __int64 v9; // rbx
  void *v10; // rdi
  size_t v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // edx

  v4 = *((unsigned int *)a1 + 2);
  if ( (unsigned int)v4 > 0x40 )
  {
    v9 = (unsigned __int64)(v4 + 63) >> 6;
    v10 = (void *)sub_2207820(8 * v9);
    if ( v10 )
    {
      v11 = 8;
      if ( (__int64)(v9 - 2) >= -1 )
        v11 = 8 * v9;
      v10 = memset(v10, 0, v11);
    }
    v12 = *((unsigned int *)a1 + 2);
    *a1 = (unsigned __int64)v10;
    v4 = v12;
    v13 = (unsigned __int64)(v12 + 63) >> 6;
    if ( a3 < v13 )
      v13 = a3;
    memcpy(v10, a2, 8 * v13);
  }
  else
  {
    *a1 = *a2;
  }
  v5 = *a1;
  result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
  if ( (_DWORD)v4 )
  {
    if ( (unsigned int)v4 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v4 + 63) >> 6) - 1;
      *(_QWORD *)(v5 + 8 * v7) &= result;
      return result;
    }
  }
  else
  {
    result = 0;
  }
  result &= v5;
  *a1 = result;
  return result;
}
