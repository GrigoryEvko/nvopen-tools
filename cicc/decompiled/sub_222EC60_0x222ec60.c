// Function: sub_222EC60
// Address: 0x222ec60
//
void *__fastcall sub_222EC60(_BYTE *src, __int64 a2)
{
  size_t v2; // rbp
  __int64 v3; // rax
  __int64 v4; // rbx
  void *v5; // r8

  v2 = a2 - (_QWORD)src;
  v3 = sub_22153F0(a2 - (_QWORD)src, 0);
  v4 = v3;
  v5 = (void *)(v3 + 24);
  if ( a2 - (_QWORD)src == 1 )
  {
    *(_BYTE *)(v3 + 24) = *src;
    if ( (_UNKNOWN *)v3 == &unk_4FD67C0 )
      return v5;
  }
  else
  {
    v5 = memcpy((void *)(v3 + 24), src, v2);
    if ( (_UNKNOWN *)v4 == &unk_4FD67C0 )
      return v5;
  }
  *(_DWORD *)(v4 + 16) = 0;
  *(_QWORD *)v4 = v2;
  *(_BYTE *)(v4 + v2 + 24) = 0;
  return v5;
}
