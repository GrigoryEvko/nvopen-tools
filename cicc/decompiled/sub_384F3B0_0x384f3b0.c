// Function: sub_384F3B0
// Address: 0x384f3b0
//
unsigned __int64 __fastcall sub_384F3B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // rsi
  unsigned __int64 result; // rax
  int v10; // r8d
  char v11; // dl
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 *v14; // rcx

  v8 = *a2;
  result = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) != result )
    goto LABEL_2;
  v12 = *(unsigned int *)(a1 + 28);
  v13 = (__int64 *)(result + 8 * v12);
  v10 = v12;
  if ( (__int64 *)result == v13 )
  {
LABEL_16:
    if ( (unsigned int)v12 < *(_DWORD *)(a1 + 24) )
    {
      v10 = v12 + 1;
      *(_DWORD *)(a1 + 28) = v12 + 1;
      *v13 = v8;
      ++*(_QWORD *)a1;
LABEL_6:
      result = *(unsigned int *)(a1 + 176);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 180) )
      {
        sub_16CD150(a1 + 168, (const void *)(a1 + 184), 0, 8, v10, a6);
        result = *(unsigned int *)(a1 + 176);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8 * result) = *a2;
      ++*(_DWORD *)(a1 + 176);
      return result;
    }
LABEL_2:
    result = (unsigned __int64)sub_16CCBA0(a1, v8);
    if ( !v11 )
      return result;
    goto LABEL_6;
  }
  v14 = 0;
  while ( v8 != *(_QWORD *)result )
  {
    if ( *(_QWORD *)result == -2 )
      v14 = (__int64 *)result;
    result += 8LL;
    if ( v13 == (__int64 *)result )
    {
      if ( !v14 )
        goto LABEL_16;
      *v14 = v8;
      --*(_DWORD *)(a1 + 32);
      ++*(_QWORD *)a1;
      goto LABEL_6;
    }
  }
  return result;
}
