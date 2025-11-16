// Function: sub_9BAA40
// Address: 0x9baa40
//
_BYTE *__fastcall sub_9BAA40(_BYTE *a1)
{
  _BYTE *v1; // r12
  __int64 *v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rax

  v1 = a1;
  if ( (a1[8] & 1) != 0 )
    return v1;
  v1 = *(_BYTE **)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1) != 0 )
    return v1;
  v4 = *(__int64 **)v1;
  if ( (*(_BYTE *)(*(_QWORD *)v1 + 8LL) & 1) != 0 )
  {
    v5 = *(_QWORD *)v1;
  }
  else
  {
    v6 = *v4;
    if ( (*(_BYTE *)(*v4 + 8) & 1) == 0 )
    {
      v7 = sub_9BAA40();
      *v4 = v7;
      v6 = v7;
    }
    *(_QWORD *)v1 = v6;
    v5 = v6;
  }
  *(_QWORD *)a1 = v5;
  return (_BYTE *)v5;
}
