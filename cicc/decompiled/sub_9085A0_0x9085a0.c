// Function: sub_9085A0
// Address: 0x9085a0
//
void __fastcall sub_9085A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  const char *v6; // r15
  char *v7; // rax
  size_t v8; // r9
  char *v9; // rdx
  char *v10; // r15
  size_t v11; // rdx
  char *v12; // rax
  char *v13; // rdi
  size_t n; // [rsp+0h] [rbp-D0h]
  int v15; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+18h] [rbp-B8h]
  char *s[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v19[18]; // [rsp+40h] [rbp-90h] BYREF

  v15 = 0;
  v5 = sub_2241E40();
  v6 = *(const char **)(a3 + 24);
  v16 = v5;
  s[0] = (char *)v18;
  if ( !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v7 = (char *)strlen(v6);
  v19[0] = v7;
  v8 = (size_t)v7;
  if ( (unsigned __int64)v7 > 0xF )
  {
    n = (size_t)v7;
    v12 = (char *)sub_22409D0(s, v19, 0);
    v8 = n;
    s[0] = v12;
    v13 = v12;
    v18[0] = v19[0];
  }
  else
  {
    if ( v7 == (char *)1 )
    {
      LOBYTE(v18[0]) = *v6;
      v9 = (char *)v18;
      goto LABEL_5;
    }
    if ( !v7 )
    {
      v9 = (char *)v18;
      goto LABEL_5;
    }
    v13 = (char *)v18;
  }
  memcpy(v13, v6, v8);
  v7 = (char *)v19[0];
  v9 = s[0];
LABEL_5:
  s[1] = v7;
  v7[(_QWORD)v9] = 0;
  v10 = s[0];
  v11 = 0;
  if ( s[0] )
    v11 = strlen(s[0]);
  sub_CB7060(v19, v10, v11, &v15, 0);
  sub_CB6200(v19, a1, a2);
  sub_CB5B00(v19);
  if ( (_QWORD *)s[0] != v18 )
    j_j___libc_free_0(s[0], v18[0] + 1LL);
}
