// Function: sub_D84780
// Address: 0xd84780
//
void __fastcall sub_D84780(__int64 *a1)
{
  _BYTE *v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdi
  _BYTE *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rdi

  if ( a1[1] )
    return;
  v2 = (_BYTE *)sub_BAA6A0(*a1, 1);
  if ( v2 )
  {
    v3 = sub_BC9AA0(v2);
    v4 = a1[1];
    a1[1] = v3;
    if ( !v4 )
      goto LABEL_8;
    v5 = *(_QWORD *)(v4 + 8);
    if ( v5 )
      j_j___libc_free_0(v5, *(_QWORD *)(v4 + 24) - v5);
    j_j___libc_free_0(v4, 88);
  }
  v3 = a1[1];
LABEL_8:
  if ( v3 )
  {
LABEL_9:
    sub_D84620((__int64)a1);
    return;
  }
  v6 = (_BYTE *)sub_BAA6A0(*a1, 0);
  if ( v6 )
  {
    v7 = sub_BC9AA0(v6);
    v8 = a1[1];
    a1[1] = v7;
    if ( !v8 )
      goto LABEL_16;
    v9 = *(_QWORD *)(v8 + 8);
    if ( v9 )
      j_j___libc_free_0(v9, *(_QWORD *)(v8 + 24) - v9);
    j_j___libc_free_0(v8, 88);
  }
  v7 = a1[1];
LABEL_16:
  if ( v7 )
    goto LABEL_9;
}
