// Function: sub_2216B70
// Address: 0x2216b70
//
void *__fastcall sub_2216B70(wchar_t *s2, wchar_t *a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r14

  if ( s2 == a2 )
    return &unk_4FD67F8;
  if ( !s2 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v2 = (char *)a2 - (char *)s2;
  v3 = a2 - s2;
  v4 = sub_2216040(v3, 0);
  v5 = v4;
  v6 = v4 + 24;
  if ( v3 == 1 )
  {
    *(_DWORD *)(v4 + 24) = *s2;
  }
  else if ( v3 )
  {
    wmemcpy((wchar_t *)(v4 + 24), s2, v2 >> 2);
  }
  if ( (_UNKNOWN *)v5 != &unk_4FD67E0 )
  {
    *(_DWORD *)(v5 + 16) = 0;
    *(_QWORD *)v5 = v3;
    *(_DWORD *)(v5 + v2 + 24) = 0;
  }
  return (void *)v6;
}
