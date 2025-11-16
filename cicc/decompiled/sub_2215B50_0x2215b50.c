// Function: sub_2215B50
// Address: 0x2215b50
//
__int64 *__fastcall sub_2215B50(__int64 *a1, _QWORD *a2)
{
  size_t v3; // r13
  size_t v4; // rbp
  _BYTE *v5; // rsi
  _BYTE *v6; // rdi
  __int64 v7; // rax

  v3 = *(_QWORD *)(*a2 - 24LL);
  if ( v3 )
  {
    v4 = v3 + *(_QWORD *)(*a1 - 24);
    if ( v4 > *(_QWORD *)(*a1 - 16) || *(int *)(*a1 - 8) > 0 )
      sub_2215AB0(a1, v3 + *(_QWORD *)(*a1 - 24));
    v5 = (_BYTE *)*a2;
    v6 = (_BYTE *)(*(_QWORD *)(*a1 - 24) + *a1);
    if ( v3 == 1 )
      *v6 = *v5;
    else
      memcpy(v6, v5, v3);
    v7 = *a1;
    if ( (_UNKNOWN *)(*a1 - 24) != &unk_4FD67C0 )
    {
      *(_DWORD *)(v7 - 8) = 0;
      *(_QWORD *)(v7 - 24) = v4;
      *(_BYTE *)(v7 + v4) = 0;
    }
  }
  return a1;
}
