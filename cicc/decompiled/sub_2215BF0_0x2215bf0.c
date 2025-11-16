// Function: sub_2215BF0
// Address: 0x2215bf0
//
__int64 *__fastcall sub_2215BF0(__int64 *a1, _BYTE *a2, size_t a3)
{
  unsigned __int64 v4; // rax
  __int64 v6; // rbx
  _BYTE *v7; // rbp
  unsigned __int64 v8; // rbx
  _BYTE *v9; // rdi
  __int64 v10; // rax
  _BYTE *v12; // rbp

  if ( !a3 )
    return a1;
  v4 = *a1;
  v6 = *(_QWORD *)(*a1 - 24);
  if ( a3 > 0x3FFFFFFFFFFFFFF9LL - v6 )
    sub_4262D8((__int64)"basic_string::append");
  v7 = a2;
  v8 = a3 + v6;
  if ( v8 <= *(_QWORD *)(v4 - 16) )
  {
    if ( *(int *)(v4 - 8) <= 0 )
      goto LABEL_7;
    v4 = *a1;
  }
  if ( v4 <= (unsigned __int64)a2 && (unsigned __int64)a2 <= v4 + *(_QWORD *)(v4 - 24) )
  {
    v12 = &a2[-v4];
    sub_2215AB0(a1, v8);
    v7 = &v12[*a1];
    v9 = (_BYTE *)(*(_QWORD *)(*a1 - 24) + *a1);
    if ( a3 != 1 )
      goto LABEL_8;
    goto LABEL_13;
  }
  sub_2215AB0(a1, v8);
LABEL_7:
  v9 = (_BYTE *)(*(_QWORD *)(*a1 - 24) + *a1);
  if ( a3 != 1 )
  {
LABEL_8:
    memcpy(v9, v7, a3);
    goto LABEL_9;
  }
LABEL_13:
  *v9 = *v7;
LABEL_9:
  v10 = *a1;
  if ( (_UNKNOWN *)(*a1 - 24) != &unk_4FD67C0 )
  {
    *(_DWORD *)(v10 - 8) = 0;
    *(_QWORD *)(v10 - 24) = v8;
    *(_BYTE *)(v10 + v8) = 0;
  }
  return a1;
}
