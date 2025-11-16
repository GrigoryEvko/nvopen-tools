// Function: sub_21BCA50
// Address: 0x21bca50
//
__int64 *__fastcall sub_21BCA50(__int64 *a1, char *a2, int a3)
{
  _QWORD *v3; // rax
  __int64 v5; // r15
  char *v6; // rbx
  unsigned __int64 v7; // rdx
  char *v8; // rax
  char *v9; // r15
  char v10; // si
  void *v11; // rdi
  _BYTE *v12; // r15
  size_t v13; // r13
  __int64 v15; // rax
  size_t v16; // [rsp+8h] [rbp-88h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-70h] BYREF
  void *v19; // [rsp+30h] [rbp-60h] BYREF
  char *v20; // [rsp+38h] [rbp-58h]
  unsigned __int64 v21; // [rsp+40h] [rbp-50h]
  char *v22; // [rsp+48h] [rbp-48h]
  int v23; // [rsp+50h] [rbp-40h]
  _QWORD *v24; // [rsp+58h] [rbp-38h]

  v3 = v17;
  v17[0] = v18;
  v17[1] = 0;
  LOBYTE(v18[0]) = 0;
  v23 = 1;
  v22 = 0;
  v21 = 0;
  v20 = 0;
  v19 = &unk_49EFBE0;
  v24 = v17;
  if ( !a3 )
    goto LABEL_14;
  v5 = (unsigned int)(a3 - 1);
  v6 = a2;
  v7 = 0;
  v8 = 0;
  v9 = &a2[v5];
  while ( 1 )
  {
    v10 = *v6;
    if ( (unsigned __int8)(*v6 - 45) <= 1u )
      break;
    if ( (unsigned __int64)v8 >= v7 )
    {
      sub_16E7DE0((__int64)&v19, v10);
    }
    else
    {
      v22 = v8 + 1;
      *v8 = v10;
    }
    v8 = v22;
LABEL_6:
    if ( v9 == v6 )
      goto LABEL_11;
LABEL_7:
    v7 = v21;
    ++v6;
  }
  if ( v7 - (unsigned __int64)v8 <= 2 )
  {
    sub_16E7EE0((__int64)&v19, "_$_", 3u);
    v8 = v22;
    goto LABEL_6;
  }
  v8[2] = 95;
  *(_WORD *)v8 = 9311;
  v8 = v22 + 3;
  v22 += 3;
  if ( v9 != v6 )
    goto LABEL_7;
LABEL_11:
  if ( v20 != v8 )
    sub_16E7BA0((__int64 *)&v19);
  v3 = v24;
LABEL_14:
  v11 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  v12 = (_BYTE *)*v3;
  v13 = v3[1];
  if ( v13 + *v3 && !v12 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v16 = v3[1];
  if ( v13 > 0xF )
  {
    v15 = sub_22409D0(a1, &v16, 0);
    *a1 = v15;
    v11 = (void *)v15;
    a1[2] = v16;
    goto LABEL_27;
  }
  if ( v13 == 1 )
  {
    *((_BYTE *)a1 + 16) = *v12;
    goto LABEL_19;
  }
  if ( v13 )
  {
LABEL_27:
    memcpy(v11, v12, v13);
    v13 = v16;
    v11 = (void *)*a1;
  }
LABEL_19:
  a1[1] = v13;
  *((_BYTE *)v11 + v13) = 0;
  sub_16E7BC0((__int64 *)&v19);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0], v18[0] + 1LL);
  return a1;
}
