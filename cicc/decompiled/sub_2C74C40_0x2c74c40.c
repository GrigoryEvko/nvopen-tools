// Function: sub_2C74C40
// Address: 0x2c74c40
//
void __fastcall sub_2C74C40(unsigned __int64 *a1, _BYTE *a2, void *a3)
{
  char v4; // al
  char *v5; // r15
  unsigned int v6; // eax
  __int64 v7; // rdx
  size_t v8; // rax
  size_t v9; // r8
  _QWORD *v10; // rdx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  size_t n; // [rsp+8h] [rbp-D8h]
  unsigned int v17; // [rsp+10h] [rbp-D0h]
  __int64 v18; // [rsp+18h] [rbp-C8h]
  __int64 v19; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-B8h] BYREF
  unsigned __int64 v21; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+38h] [rbp-A8h] BYREF
  char *s; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v24; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v25[2]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v26[2]; // [rsp+70h] [rbp-70h] BYREF
  void *v27[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v28; // [rsp+A0h] [rbp-40h]

  v4 = a2[32];
  if ( v4 )
  {
    if ( v4 == 1 )
    {
      v27[0] = "DataLayoutError: ";
      v28 = 259;
    }
    else
    {
      if ( a2[33] == 1 )
      {
        a3 = (void *)*((_QWORD *)a2 + 1);
        a2 = *(_BYTE **)a2;
      }
      else
      {
        v4 = 2;
      }
      v27[2] = a2;
      v27[0] = "DataLayoutError: ";
      v27[3] = a3;
      LOBYTE(v28) = 3;
      HIBYTE(v28) = v4;
    }
  }
  else
  {
    v28 = 256;
  }
  sub_CA0F50((__int64 *)&s, v27);
  v5 = s;
  v6 = sub_C63BB0();
  v25[0] = (unsigned __int64)v26;
  v18 = v7;
  v17 = v6;
  if ( !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v8 = strlen(v5);
  v22 = v8;
  v9 = v8;
  if ( v8 > 0xF )
  {
    n = v8;
    v14 = sub_22409D0((__int64)v25, (unsigned __int64 *)&v22, 0);
    v9 = n;
    v25[0] = v14;
    v15 = (_QWORD *)v14;
    v26[0] = v22;
  }
  else
  {
    if ( v8 == 1 )
    {
      LOBYTE(v26[0]) = *v5;
      v10 = v26;
      goto LABEL_11;
    }
    if ( !v8 )
    {
      v10 = v26;
      goto LABEL_11;
    }
    v15 = v26;
  }
  memcpy(v15, v5, v9);
  v8 = v22;
  v10 = (_QWORD *)v25[0];
LABEL_11:
  v25[1] = v8;
  *((_BYTE *)v10 + v8) = 0;
  sub_C63F00(&v20, (__int64)v25, v17, v18);
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  v11 = v20;
  v12 = *a1;
  *a1 = 0;
  v20 = 0;
  v25[0] = v11 | 1;
  v22 = v12 | 1;
  v19 = 0;
  sub_9CDB40(&v21, (unsigned __int64 *)&v22, v25);
  if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v22, (__int64)&v22);
  if ( (v25[0] & 1) != 0 || (v25[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(v25, (__int64)&v22);
  v13 = *a1;
  if ( (*a1 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(a1, (__int64)&v22);
  *a1 = v21 | v13 | 1;
  if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v19, (__int64)&v22);
  if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v20, (__int64)&v22);
  if ( s != (char *)&v24 )
    j_j___libc_free_0((unsigned __int64)s);
}
