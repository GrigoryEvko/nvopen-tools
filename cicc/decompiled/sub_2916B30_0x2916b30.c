// Function: sub_2916B30
// Address: 0x2916b30
//
unsigned __int64 __fastcall sub_2916B30(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  char v7; // dl
  __int64 v8; // r12

  result = *(_QWORD *)(a1 + 376);
  if ( *(_BYTE *)result )
    return result;
  if ( !*(_BYTE *)(a1 + 572) )
    goto LABEL_8;
  result = *(_QWORD *)(a1 + 552);
  a4 = *(unsigned int *)(a1 + 564);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_12:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 560) )
    {
      *(_DWORD *)(a1 + 564) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 544);
LABEL_9:
      v8 = *(_QWORD *)(a1 + 376);
      result = *(unsigned int *)(v8 + 240);
      if ( result + 1 > *(unsigned int *)(v8 + 244) )
      {
        sub_C8D5F0(v8 + 232, (const void *)(v8 + 248), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(v8 + 240);
      }
      *(_QWORD *)(*(_QWORD *)(v8 + 232) + 8 * result) = a2;
      ++*(_DWORD *)(v8 + 240);
      return result;
    }
LABEL_8:
    result = (unsigned __int64)sub_C8CC70(a1 + 544, a2, (__int64)a3, a4, a5, a6);
    if ( !v7 )
      return result;
    goto LABEL_9;
  }
  while ( a2 != *(_QWORD *)result )
  {
    result += 8LL;
    if ( a3 == (__int64 *)result )
      goto LABEL_12;
  }
  return result;
}
