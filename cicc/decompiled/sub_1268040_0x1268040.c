// Function: sub_1268040
// Address: 0x1268040
//
void __fastcall sub_1268040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  const char *v8; // r15
  char *v9; // rax
  size_t v10; // r9
  char *v11; // rdx
  char *v12; // r15
  size_t v13; // rdx
  char *v14; // rax
  char *v15; // rdi
  size_t n; // [rsp+0h] [rbp-C0h]
  int v17; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-A8h]
  char *s[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v21[16]; // [rsp+40h] [rbp-80h] BYREF

  v17 = 0;
  v7 = sub_2241E40(a1, a2, a3, a4, a5);
  v8 = *(const char **)(a3 + 24);
  v18 = v7;
  s[0] = (char *)v20;
  if ( !v8 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v9 = (char *)strlen(v8);
  v21[0] = v9;
  v10 = (size_t)v9;
  if ( (unsigned __int64)v9 > 0xF )
  {
    n = (size_t)v9;
    v14 = (char *)sub_22409D0(s, v21, 0);
    v10 = n;
    s[0] = v14;
    v15 = v14;
    v20[0] = v21[0];
  }
  else
  {
    if ( v9 == (char *)1 )
    {
      LOBYTE(v20[0]) = *v8;
      v11 = (char *)v20;
      goto LABEL_5;
    }
    if ( !v9 )
    {
      v11 = (char *)v20;
      goto LABEL_5;
    }
    v15 = (char *)v20;
  }
  memcpy(v15, v8, v10);
  v9 = (char *)v21[0];
  v11 = s[0];
LABEL_5:
  s[1] = v9;
  v9[(_QWORD)v11] = 0;
  v12 = s[0];
  v13 = 0;
  if ( s[0] )
    v13 = strlen(s[0]);
  sub_16E8AF0(v21, v12, v13, &v17, 0);
  sub_16E7EE0(v21, a1, a2);
  sub_16E7C30(v21);
  if ( (_QWORD *)s[0] != v20 )
    j_j___libc_free_0(s[0], v20[0] + 1LL);
}
