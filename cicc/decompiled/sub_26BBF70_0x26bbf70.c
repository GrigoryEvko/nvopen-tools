// Function: sub_26BBF70
// Address: 0x26bbf70
//
unsigned __int64 __fastcall sub_26BBF70(unsigned __int64 *a1, unsigned __int64 **a2, _QWORD **a3)
{
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // rax
  _QWORD *v10; // r10
  unsigned __int64 v11; // r11
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // r12

  v5 = (_QWORD *)sub_22077B0(0x18u);
  v6 = (unsigned __int64)v5;
  if ( v5 )
    *v5 = 0;
  v7 = a1[1];
  v8 = **a2;
  v5[2] = **a3;
  v9 = *a1;
  *(_QWORD *)(v6 + 8) = v8;
  v10 = *(_QWORD **)(v9 + 8 * (v8 % v7));
  v11 = v8 % v7;
  if ( !v10 )
    return sub_26BAF80(a1, v11, v8, v6, 1);
  v12 = (_QWORD *)*v10;
  if ( *(_QWORD *)(*v10 + 8LL) != v8 )
  {
    while ( 1 )
    {
      v13 = (_QWORD *)*v12;
      if ( !*v12 )
        break;
      v10 = v12;
      if ( v11 != v13[1] % v7 )
        break;
      v12 = (_QWORD *)*v12;
      if ( v13[1] == v8 )
        goto LABEL_8;
    }
    return sub_26BAF80(a1, v11, v8, v6, 1);
  }
LABEL_8:
  if ( !*v10 )
    return sub_26BAF80(a1, v11, v8, v6, 1);
  v14 = *v10;
  j_j___libc_free_0(v6);
  return v14;
}
