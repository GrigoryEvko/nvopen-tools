// Function: sub_2257250
// Address: 0x2257250
//
void *__fastcall sub_2257250(_QWORD *a1, const char *a2)
{
  size_t v2; // rax
  size_t v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  void *v6; // rcx
  void *result; // rax

  *a1 = off_4A080E0;
  if ( !a2 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v2 = strlen(a2);
  v3 = v2;
  if ( v2 )
  {
    v4 = sub_22153F0(v2, 0);
    v5 = v4;
    v6 = (void *)(v4 + 24);
    if ( v3 == 1 )
    {
      result = (void *)*(unsigned __int8 *)a2;
      *(_BYTE *)(v5 + 24) = (_BYTE)result;
    }
    else
    {
      result = memcpy((void *)(v4 + 24), a2, v3);
      v6 = result;
    }
    if ( (_UNKNOWN *)v5 != &unk_4FD67C0 )
    {
      *(_DWORD *)(v5 + 16) = 0;
      *(_QWORD *)v5 = v3;
      *(_BYTE *)(v5 + v3 + 24) = 0;
    }
    a1[1] = v6;
  }
  else
  {
    a1[1] = &unk_4FD67D8;
    return &unk_4FD67C0;
  }
  return result;
}
