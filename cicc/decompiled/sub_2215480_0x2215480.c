// Function: sub_2215480
// Address: 0x2215480
//
void *__fastcall sub_2215480(size_t n, char a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  void *v5; // r8

  if ( !n )
    return &unk_4FD67D8;
  v3 = sub_22153F0(n, 0);
  v4 = v3;
  v5 = (void *)(v3 + 24);
  if ( n == 1 )
    *(_BYTE *)(v3 + 24) = a2;
  else
    v5 = memset((void *)(v3 + 24), a2, n);
  if ( (_UNKNOWN *)v4 != &unk_4FD67C0 )
  {
    *(_DWORD *)(v4 + 16) = 0;
    *(_QWORD *)v4 = n;
    *(_BYTE *)(v4 + n + 24) = 0;
  }
  return v5;
}
