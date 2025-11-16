// Function: sub_31F5640
// Address: 0x31f5640
//
__int64 __fastcall sub_31F5640(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, char *a5)
{
  __int64 v5; // rdx
  __int64 v7; // rbx
  char *v8; // rax
  _BYTE *v9; // r9
  char *v10; // r8
  char *v11; // rax
  char *v12; // rdi
  char *v13; // rax
  char *v14; // rdx
  char *v16; // rdi
  _BYTE *src; // [rsp+10h] [rbp-80h]
  size_t n; // [rsp+18h] [rbp-78h]
  char *v21; // [rsp+20h] [rbp-70h] BYREF
  char *v22; // [rsp+28h] [rbp-68h]
  _QWORD v23[2]; // [rsp+30h] [rbp-60h] BYREF
  char *v24; // [rsp+40h] [rbp-50h] BYREF
  char *v25; // [rsp+48h] [rbp-48h]
  _QWORD v26[8]; // [rsp+50h] [rbp-40h] BYREF

  v5 = 16 * a3;
  v7 = a2 + v5;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( a2 != a2 + v5 )
  {
    while ( 1 )
    {
      v9 = *(_BYTE **)(v7 - 16);
      v10 = *(char **)(v7 - 8);
      v21 = (char *)v23;
      if ( &v9[(_QWORD)v10] && !v9 )
        goto LABEL_29;
      v24 = v10;
      if ( (unsigned __int64)v10 > 0xF )
        break;
      if ( v10 == (char *)1 )
      {
        LOBYTE(v23[0]) = *v9;
        v8 = (char *)v23;
      }
      else
      {
        if ( v10 )
        {
          v12 = (char *)v23;
          goto LABEL_15;
        }
        v8 = (char *)v23;
      }
LABEL_4:
      v22 = v10;
      v10[(_QWORD)v8] = 0;
      sub_2241490((unsigned __int64 *)a1, v21, (size_t)v22);
      if ( v21 != (char *)v23 )
        j_j___libc_free_0((unsigned __int64)v21);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 1 )
        sub_4262D8((__int64)"basic_string::append");
      v7 -= 16;
      sub_2241490((unsigned __int64 *)a1, "::", 2u);
      if ( a2 == v7 )
        goto LABEL_16;
    }
    src = v9;
    n = (size_t)v10;
    v11 = (char *)sub_22409D0((__int64)&v21, (unsigned __int64 *)&v24, 0);
    v10 = (char *)n;
    v9 = src;
    v21 = v11;
    v12 = v11;
    v23[0] = v24;
LABEL_15:
    memcpy(v12, v9, (size_t)v10);
    v10 = v24;
    v8 = v21;
    goto LABEL_4;
  }
LABEL_16:
  v24 = (char *)v26;
  v13 = a5;
  if ( &a4[(_QWORD)a5] && !a4 )
LABEL_29:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v21 = a5;
  if ( (unsigned __int64)a5 > 0xF )
  {
    v24 = (char *)sub_22409D0((__int64)&v24, (unsigned __int64 *)&v21, 0);
    v16 = v24;
    v26[0] = v21;
  }
  else
  {
    if ( a5 == (char *)1 )
    {
      LOBYTE(v26[0]) = *a4;
      v14 = (char *)v26;
      goto LABEL_21;
    }
    if ( !a5 )
    {
      v14 = (char *)v26;
      goto LABEL_21;
    }
    v16 = (char *)v26;
  }
  memcpy(v16, a4, (size_t)a5);
  v13 = v21;
  v14 = v24;
LABEL_21:
  v25 = v13;
  v13[(_QWORD)v14] = 0;
  sub_2241490((unsigned __int64 *)a1, v24, (size_t)v25);
  if ( v24 != (char *)v26 )
    j_j___libc_free_0((unsigned __int64)v24);
  return a1;
}
