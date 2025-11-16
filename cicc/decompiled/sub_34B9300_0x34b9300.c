// Function: sub_34B9300
// Address: 0x34b9300
//
__int64 __fastcall sub_34B9300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char *a5)
{
  char *v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  char *v11; // r15
  int i; // esi
  unsigned int v13; // r12d
  char v16; // [rsp+17h] [rbp-F9h] BYREF
  __int64 v17; // [rsp+18h] [rbp-F8h] BYREF
  _BYTE v18[8]; // [rsp+20h] [rbp-F0h] BYREF
  char *v19; // [rsp+28h] [rbp-E8h]
  char v20; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v21; // [rsp+80h] [rbp-90h] BYREF
  char *v22; // [rsp+88h] [rbp-88h]
  char v23; // [rsp+98h] [rbp-78h] BYREF

  v5 = a5;
  if ( !a5 )
    v5 = &v16;
  *v5 = 1;
  v21 = *(_QWORD *)(a1 + 120);
  v6 = sub_A74610(&v21);
  v7 = sub_B2BE50(a1);
  sub_A74940((__int64)v18, v7, v6);
  v17 = *(_QWORD *)(a2 + 72);
  v8 = sub_A74610(&v17);
  v9 = sub_B2BE50(a1);
  v10 = v8;
  v11 = (char *)&unk_44E2400;
  sub_A74940((__int64)&v21, v9, v10);
  for ( i = 86; ; i = *(_DWORD *)v11 )
  {
    v11 += 4;
    sub_A77390((__int64)v18, i);
    sub_A77390((__int64)&v21, *((_DWORD *)v11 - 1));
    if ( "Basic Block Path Cloning" == v11 )
      break;
  }
  if ( sub_A75040((__int64)v18, 79) )
  {
    if ( !sub_A75040((__int64)&v21, 79) )
    {
LABEL_8:
      v13 = 0;
      goto LABEL_12;
    }
    *v5 = 0;
    sub_A77390((__int64)v18, 79);
    sub_A77390((__int64)&v21, 79);
    if ( !*(_QWORD *)(a2 + 16) )
    {
LABEL_18:
      sub_A77390((__int64)&v21, 54);
      sub_A77390((__int64)&v21, 79);
    }
  }
  else
  {
    if ( sub_A75040((__int64)v18, 54) )
    {
      if ( !sub_A75040((__int64)&v21, 54) )
        goto LABEL_8;
      *v5 = 0;
      sub_A77390((__int64)v18, 54);
      sub_A77390((__int64)&v21, 54);
    }
    if ( !*(_QWORD *)(a2 + 16) )
      goto LABEL_18;
  }
  v13 = sub_A75080((__int64)v18, (__int64)&v21);
LABEL_12:
  if ( v22 != &v23 )
    _libc_free((unsigned __int64)v22);
  if ( v19 != &v20 )
    _libc_free((unsigned __int64)v19);
  return v13;
}
