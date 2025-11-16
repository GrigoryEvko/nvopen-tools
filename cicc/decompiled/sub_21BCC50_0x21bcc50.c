// Function: sub_21BCC50
// Address: 0x21bcc50
//
__int64 __fastcall sub_21BCC50(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  char *v3; // rax
  int v4; // edx
  __int64 j; // rbx
  char *v6; // rax
  int v7; // edx
  __int64 v9[2]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v10; // [rsp+10h] [rbp-60h]
  __int64 v11[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v12[8]; // [rsp+30h] [rbp-40h] BYREF

  for ( i = *(_QWORD *)(a2 + 16); a2 + 8 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( (*(_BYTE *)(i - 24) & 0xFu) - 7 <= 1 )
    {
      v3 = (char *)sub_1649960(i - 56);
      sub_21BCA50(v11, v3, v4);
      v9[0] = (__int64)v11;
      v10 = 260;
      sub_164B780(i - 56, v9);
      if ( (_QWORD *)v11[0] != v12 )
        j_j___libc_free_0(v11[0], v12[0] + 1LL);
    }
  }
  for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
  {
    if ( !j )
      BUG();
    if ( (*(_BYTE *)(j - 24) & 0xFu) - 7 <= 1 )
    {
      v6 = (char *)sub_1649960(j - 56);
      sub_21BCA50(v11, v6, v7);
      v9[0] = (__int64)v11;
      v10 = 260;
      sub_164B780(j - 56, v9);
      if ( (_QWORD *)v11[0] != v12 )
        j_j___libc_free_0(v11[0], v12[0] + 1LL);
    }
  }
  return 1;
}
