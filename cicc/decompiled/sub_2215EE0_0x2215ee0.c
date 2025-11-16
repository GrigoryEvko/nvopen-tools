// Function: sub_2215EE0
// Address: 0x2215ee0
//
void *__fastcall sub_2215EE0(_BYTE *src, _BYTE *a2)
{
  size_t v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r12
  void *v5; // r8

  if ( src == a2 )
    return &unk_4FD67D8;
  if ( !src )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v2 = a2 - src;
  v3 = sub_22153F0(a2 - src, 0);
  v4 = v3;
  v5 = (void *)(v3 + 24);
  if ( a2 - src == 1 )
    *(_BYTE *)(v3 + 24) = *src;
  else
    v5 = memcpy((void *)(v3 + 24), src, v2);
  if ( (_UNKNOWN *)v4 != &unk_4FD67C0 )
  {
    *(_DWORD *)(v4 + 16) = 0;
    *(_QWORD *)v4 = v2;
    *(_BYTE *)(v4 + v2 + 24) = 0;
  }
  return v5;
}
