// Function: sub_11DA720
// Address: 0x11da720
//
__int64 __fastcall sub_11DA720(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 i; // rbx
  _BYTE *v6; // rax
  _BYTE *v8; // rdi
  bool v9; // al
  char v10; // bl
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]
  unsigned __int64 v13; // [rsp+8h] [rbp-38h]
  unsigned __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v6 = *(_BYTE **)(i + 24);
    if ( *v6 != 82 )
      return 0;
    v8 = (_BYTE *)*((_QWORD *)v6 - 4);
    v12 = a4;
    v13 = a3;
    if ( *v8 > 0x15u )
      return 0;
    v9 = sub_AC30F0((__int64)v8);
    a3 = v13;
    a4 = v12;
    if ( !v9 )
      return 0;
  }
  v14 = a3;
  v15 = 64;
  v10 = sub_D30550(a2, 0, &v14, a4, 0, 0, 0, 0);
  if ( v15 > 0x40 )
  {
    if ( v14 )
      j_j___libc_free_0_0(v14);
  }
  if ( !v10 )
    return 0;
  v11 = sub_B43CB0(a1);
  return (unsigned int)sub_B2D610(v11, 59) ^ 1;
}
