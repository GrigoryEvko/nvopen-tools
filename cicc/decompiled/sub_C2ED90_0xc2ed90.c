// Function: sub_C2ED90
// Address: 0xc2ed90
//
_QWORD *__fastcall sub_C2ED90(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rax
  char *v9; // r15
  __int64 v10; // rdx
  char *v11; // rdi
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-88h]
  _BYTE v15[16]; // [rsp+10h] [rbp-80h] BYREF
  char *s[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v18; // [rsp+40h] [rbp-50h] BYREF
  int v19; // [rsp+48h] [rbp-48h]
  _QWORD v20[8]; // [rsp+50h] [rbp-40h] BYREF

  sub_C88F40(v15, a3, a4, 0);
  s[0] = (char *)v17;
  s[1] = 0;
  LOBYTE(v17[0]) = 0;
  if ( (unsigned __int8)sub_C89030(v15, s) )
  {
    if ( *(_BYTE *)(a2 + 16) )
    {
      sub_C88FD0(&v18, v15);
      v13 = *(_QWORD *)a2;
      *(_QWORD *)a2 = v18;
      v18 = (_QWORD *)v13;
      LODWORD(v13) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a2 + 8) = v19;
      v19 = v13;
      sub_C88FF0(&v18);
    }
    else
    {
      sub_C88FD0(a2, v15);
      *(_BYTE *)(a2 + 16) = 1;
    }
    v11 = s[0];
    *a1 = 1;
    if ( v11 != (char *)v17 )
LABEL_7:
      j_j___libc_free_0(v11, v17[0] + 1LL);
  }
  else
  {
    v8 = sub_2241E50(v15, s, v5, v6, v7);
    v9 = s[0];
    v10 = -1;
    v14 = v8;
    v18 = v20;
    if ( s[0] )
      v10 = (__int64)&v9[strlen(s[0])];
    sub_C2EC30((__int64 *)&v18, v9, v10);
    sub_C63F00(a1, &v18, 22, v14);
    if ( v18 != v20 )
      j_j___libc_free_0(v18, v20[0] + 1LL);
    v11 = s[0];
    if ( (_QWORD *)s[0] != v17 )
      goto LABEL_7;
  }
  sub_C88FF0(v15);
  return a1;
}
