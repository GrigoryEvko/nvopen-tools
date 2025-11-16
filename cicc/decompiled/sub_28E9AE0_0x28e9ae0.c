// Function: sub_28E9AE0
// Address: 0x28e9ae0
//
__int64 __fastcall sub_28E9AE0(__int64 a1, char *a2)
{
  const void *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax

  v4 = a2 + 16;
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)a1 + 16 * v5;
  if ( (const void *)v6 != v4 )
  {
    memmove(a2, v4, v6 - (_QWORD)v4);
    LODWORD(v5) = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v5 - 1);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
