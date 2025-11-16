// Function: sub_16A4EF0
// Address: 0x16a4ef0
//
unsigned __int64 __fastcall sub_16A4EF0(__int64 a1, __int64 a2, char a3)
{
  size_t v5; // r12
  void *v6; // rax
  _QWORD *v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // ecx
  unsigned __int64 result; // rax
  __int64 v13; // rdx

  v5 = 8 * (((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6);
  v6 = (void *)sub_2207820(v5);
  v7 = memset(v6, 0, v5);
  *(_QWORD *)a1 = v7;
  v8 = (unsigned __int64)v7;
  *v7 = a2;
  if ( a2 < 0 && a3 )
  {
    v9 = *(unsigned int *)(a1 + 8);
    if ( (unsigned __int64)(v9 + 63) > 0x7F )
    {
      v10 = 8;
      v11 = 1;
      do
      {
        *(_QWORD *)(v8 + v10) = -1;
        ++v11;
        v10 += 8;
        v8 = *(_QWORD *)a1;
        v9 = *(unsigned int *)(a1 + 8);
      }
      while ( v11 < (unsigned int)((unsigned __int64)(v9 + 63) >> 6) );
    }
  }
  else
  {
    v9 = *(unsigned int *)(a1 + 8);
  }
  result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
  if ( (unsigned int)v9 > 0x40 )
  {
    v13 = (unsigned int)((unsigned __int64)(v9 + 63) >> 6) - 1;
    *(_QWORD *)(v8 + 8 * v13) &= result;
  }
  else
  {
    result &= v8;
    *(_QWORD *)a1 = result;
  }
  return result;
}
