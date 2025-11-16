// Function: sub_A5B750
// Address: 0xa5b750
//
_BYTE *__fastcall sub_A5B750(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v18[8]; // [rsp+10h] [rbp-40h] BYREF

  v17 = a3;
  if ( !a2 )
    return (_BYTE *)sub_904010(*a1, "<null operand!>");
  v3 = a1 + 5;
  sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 8), *a1);
  if ( v17 )
  {
    v8 = *a1;
    v9 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v8, 32);
    }
    else
    {
      v10 = (__int64)(v9 + 1);
      *(_QWORD *)(v8 + 32) = v9 + 1;
      *v9 = 32;
    }
    sub_A58630(a1, (__int64)&v17, v10, v5, v6, v7);
  }
  v11 = *a1;
  v12 = *(_BYTE **)(*a1 + 32);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(*a1 + 24) )
  {
    sub_CB5D20(v11, 32);
  }
  else
  {
    *(_QWORD *)(v11 + 32) = v12 + 1;
    *v12 = 32;
  }
  v13 = a1[4];
  v14 = a1[1];
  v18[1] = v3;
  v15 = *a1;
  v18[2] = v13;
  v18[0] = off_4979428;
  v18[3] = v14;
  return sub_A5A730(v15, a2, (__int64)v18);
}
