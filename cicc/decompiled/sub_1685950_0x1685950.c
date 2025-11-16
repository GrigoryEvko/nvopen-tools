// Function: sub_1685950
// Address: 0x1685950
//
_QWORD *__fastcall sub_1685950(_QWORD *src, size_t n)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // r15
  _QWORD *v5; // rax
  size_t v6; // rdx
  _QWORD *v7; // r14

  v2 = sub_1684C80((unsigned __int64)src);
  if ( !v2 )
    return (_QWORD *)sub_1688BE0(src, n);
  v3 = v2;
  sub_1684B50((pthread_mutex_t **)(*(_QWORD *)(v2 + 24) + 7128LL));
  if ( *(_BYTE *)(v3 + 40) )
    v4 = *(unsigned int *)(v3 + 48);
  else
    v4 = *(src - 2) - 32LL;
  j__pthread_mutex_unlock(*(pthread_mutex_t **)(*(_QWORD *)(v3 + 24) + 7128LL));
  v5 = sub_1685080(*(_QWORD *)(v3 + 24), n);
  v6 = v4;
  if ( n <= v4 )
    v6 = n;
  v7 = v5;
  memcpy(v5, src, v6);
  sub_16856A0(src);
  return v7;
}
