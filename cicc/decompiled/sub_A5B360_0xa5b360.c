// Function: sub_A5B360
// Address: 0xa5b360
//
_BYTE *__fastcall sub_A5B360(__int64 *a1, __int64 a2, char a3)
{
  __int64 *v3; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _QWORD v11[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a2 )
    return (_BYTE *)sub_904010(*a1, "<null operand!>");
  v3 = a1 + 5;
  if ( a3 )
  {
    sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 8), *a1);
    v9 = *a1;
    v10 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v9, 32);
    }
    else
    {
      *(_QWORD *)(v9 + 32) = v10 + 1;
      *v10 = 32;
    }
  }
  v5 = a1[4];
  v6 = a1[1];
  v11[1] = v3;
  v7 = *a1;
  v11[2] = v5;
  v11[0] = off_4979428;
  v11[3] = v6;
  return sub_A5A730(v7, a2, (__int64)v11);
}
