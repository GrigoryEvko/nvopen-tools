// Function: sub_2215CF0
// Address: 0x2215cf0
//
__int64 *__fastcall sub_2215CF0(__int64 *a1, size_t a2, char a3)
{
  __int64 v4; // rcx
  __int64 v6; // rax
  unsigned __int64 v7; // rbp
  _BYTE *v8; // rdi
  __int64 v9; // rax

  if ( a2 )
  {
    v4 = *a1;
    v6 = *(_QWORD *)(*a1 - 24);
    if ( a2 > 0x3FFFFFFFFFFFFFF9LL - v6 )
      sub_4262D8((__int64)"basic_string::append");
    v7 = a2 + v6;
    if ( a2 + v6 > *(_QWORD *)(v4 - 16) || *(int *)(v4 - 8) > 0 )
      sub_2215AB0(a1, v7);
    v8 = (_BYTE *)(*(_QWORD *)(*a1 - 24) + *a1);
    if ( a2 == 1 )
      *v8 = a3;
    else
      memset(v8, a3, a2);
    v9 = *a1;
    if ( (_UNKNOWN *)(*a1 - 24) != &unk_4FD67C0 )
    {
      *(_DWORD *)(v9 - 8) = 0;
      *(_QWORD *)(v9 - 24) = v7;
      *(_BYTE *)(v9 + v7) = 0;
    }
  }
  return a1;
}
