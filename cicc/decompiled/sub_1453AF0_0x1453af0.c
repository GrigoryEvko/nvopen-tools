// Function: sub_1453AF0
// Address: 0x1453af0
//
__int64 __fastcall sub_1453AF0(__int64 a1, char *a2)
{
  const void *v4; // rsi
  __int64 v5; // rdx
  int v6; // eax
  __int64 result; // rax

  v4 = a2 + 8;
  v5 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  v6 = *(_DWORD *)(a1 + 8);
  if ( (const void *)v5 != v4 )
  {
    memmove(a2, v4, v5 - (_QWORD)v4);
    v6 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v6 - 1);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
