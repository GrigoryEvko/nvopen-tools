// Function: sub_13D6230
// Address: 0x13d6230
//
_QWORD *__fastcall sub_13D6230(__int64 a1, char *a2, char *a3)
{
  __int64 v3; // r13
  __int64 v5; // rdx
  char *v6; // rbx
  unsigned __int64 v7; // r13
  _QWORD *result; // rax

  v3 = a3 - a2;
  v5 = *(unsigned int *)(a1 + 8);
  v6 = a2;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v5 < v7 )
  {
    sub_16CD150(a1, a1 + 16, v7 + v5, 8);
    v5 = *(unsigned int *)(a1 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)a1 + 8 * v5);
  if ( a2 != a3 )
  {
    do
    {
      if ( result )
        *result = *(_QWORD *)v6;
      v6 += 24;
      ++result;
    }
    while ( a3 != v6 );
    LODWORD(v5) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v7 + v5;
  return result;
}
