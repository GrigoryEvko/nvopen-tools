// Function: sub_2242720
// Address: 0x2242720
//
_QWORD *__fastcall sub_2242720(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // r13
  size_t v4; // rax
  size_t v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rbp
  void *v8; // rcx

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(_BYTE **)(v2 + 16);
  if ( !v3 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v4 = strlen(*(const char **)(v2 + 16));
  v5 = v4;
  if ( v4 )
  {
    v6 = sub_22153F0(v4, 0);
    v7 = v6;
    v8 = (void *)(v6 + 24);
    if ( v5 == 1 )
      *(_BYTE *)(v6 + 24) = *v3;
    else
      v8 = memcpy((void *)(v6 + 24), v3, v5);
    if ( (_UNKNOWN *)v7 != &unk_4FD67C0 )
    {
      *(_DWORD *)(v7 + 16) = 0;
      *(_QWORD *)v7 = v5;
      *(_BYTE *)(v7 + v5 + 24) = 0;
    }
  }
  else
  {
    v8 = &unk_4FD67D8;
  }
  *a1 = v8;
  return a1;
}
