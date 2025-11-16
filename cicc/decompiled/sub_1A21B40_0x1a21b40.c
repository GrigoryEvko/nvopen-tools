// Function: sub_1A21B40
// Address: 0x1a21b40
//
unsigned __int64 __fastcall sub_1A21B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  char v9; // dl
  __int64 v10; // r12
  __int64 *v11; // rsi
  unsigned int v12; // edi
  __int64 *v13; // rcx

  result = *(_QWORD *)(a1 + 552);
  if ( *(_QWORD *)(a1 + 560) != result )
    goto LABEL_2;
  v11 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 572));
  v12 = *(_DWORD *)(a1 + 572);
  if ( (__int64 *)result == v11 )
  {
LABEL_16:
    if ( v12 < *(_DWORD *)(a1 + 568) )
    {
      *(_DWORD *)(a1 + 572) = v12 + 1;
      *v11 = a2;
      ++*(_QWORD *)(a1 + 544);
LABEL_6:
      v10 = *(_QWORD *)(a1 + 376);
      result = *(unsigned int *)(v10 + 224);
      if ( (unsigned int)result >= *(_DWORD *)(v10 + 228) )
      {
        sub_16CD150(v10 + 216, (const void *)(v10 + 232), 0, 8, a5, a6);
        result = *(unsigned int *)(v10 + 224);
      }
      *(_QWORD *)(*(_QWORD *)(v10 + 216) + 8 * result) = a2;
      ++*(_DWORD *)(v10 + 224);
      return result;
    }
LABEL_2:
    result = (unsigned __int64)sub_16CCBA0(a1 + 544, a2);
    if ( !v9 )
      return result;
    goto LABEL_6;
  }
  v13 = 0;
  while ( a2 != *(_QWORD *)result )
  {
    if ( *(_QWORD *)result == -2 )
      v13 = (__int64 *)result;
    result += 8LL;
    if ( v11 == (__int64 *)result )
    {
      if ( !v13 )
        goto LABEL_16;
      *v13 = a2;
      --*(_DWORD *)(a1 + 576);
      ++*(_QWORD *)(a1 + 544);
      goto LABEL_6;
    }
  }
  return result;
}
