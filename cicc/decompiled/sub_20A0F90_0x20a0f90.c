// Function: sub_20A0F90
// Address: 0x20a0f90
//
void *__fastcall sub_20A0F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // eax
  void *result; // rax

  v5 = *(_DWORD *)(a4 + 8);
  if ( v5 > 0x40 )
  {
    memset(*(void **)a4, 0, 8 * (((unsigned __int64)v5 + 63) >> 6));
    result = (void *)*(unsigned int *)(a4 + 24);
    if ( (unsigned int)result <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a4 = 0;
    result = (void *)*(unsigned int *)(a4 + 24);
    if ( (unsigned int)result <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)(a4 + 16) = 0;
      return result;
    }
  }
  return memset(*(void **)(a4 + 16), 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
}
