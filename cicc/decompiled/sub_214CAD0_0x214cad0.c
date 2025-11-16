// Function: sub_214CAD0
// Address: 0x214cad0
//
void __fastcall sub_214CAD0(_BYTE *a1, __int64 a2)
{
  char v3; // al
  const char *v4; // rsi
  unsigned __int64 v5; // rdx
  const char *v6; // r15
  size_t v7; // r12
  unsigned __int64 v8; // rax
  const char *v9; // rdx
  char *v10; // rdi
  unsigned __int64 v11; // [rsp+8h] [rbp-78h] BYREF
  _QWORD *v12; // [rsp+10h] [rbp-70h] BYREF
  __int64 v13; // [rsp+18h] [rbp-68h]
  _QWORD v14[2]; // [rsp+20h] [rbp-60h] BYREF
  char *v15; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v16; // [rsp+38h] [rbp-48h]
  char v17[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1[32] & 0xF;
  if ( !v3 )
  {
    v4 = ".visible ";
    if ( sub_15E4F60((__int64)a1) )
      v4 = ".extern ";
    goto LABEL_4;
  }
  if ( v3 != 6 )
  {
    if ( v3 == 10 )
    {
      v4 = ".common ";
      if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8 == 1 )
      {
LABEL_4:
        sub_1263B40(a2, v4);
        return;
      }
    }
    else if ( ((v3 + 9) & 0xFu) <= 1 )
    {
      return;
    }
    v4 = ".weak ";
    goto LABEL_4;
  }
  v13 = 0;
  v12 = v14;
  LOBYTE(v14[0]) = 0;
  sub_2241490(&v12, "Error: ", 7);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v13) <= 6 )
LABEL_28:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v12, "Symbol ", 7);
  if ( (a1[23] & 0x20) != 0 )
  {
    v6 = sub_1649960((__int64)a1);
    v7 = v5;
    if ( !v6 )
    {
      v16 = 0;
      v15 = v17;
      v17[0] = 0;
      sub_2241490(&v12, v17, 0);
LABEL_21:
      if ( v15 != v17 )
        j_j___libc_free_0(v15, *(_QWORD *)v17 + 1LL);
      goto LABEL_11;
    }
    v11 = v5;
    v8 = v5;
    v15 = v17;
    if ( v5 > 0xF )
    {
      v15 = (char *)sub_22409D0(&v15, &v11, 0);
      v10 = v15;
      *(_QWORD *)v17 = v11;
    }
    else
    {
      if ( v5 == 1 )
      {
        v17[0] = *v6;
        v9 = v17;
LABEL_20:
        v16 = v8;
        v9[v8] = 0;
        sub_2241490(&v12, v15, v16);
        goto LABEL_21;
      }
      if ( !v5 )
      {
        v9 = v17;
        goto LABEL_20;
      }
      v10 = v17;
    }
    memcpy(v10, v6, v7);
    v8 = v11;
    v9 = v15;
    goto LABEL_20;
  }
LABEL_11:
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v13) <= 0x25 )
    goto LABEL_28;
  sub_2241490(&v12, "has unsupported appending linkage type", 38);
  sub_1C3EFD0((__int64)&v12, 1);
  if ( v12 != v14 )
    j_j___libc_free_0(v12, v14[0] + 1LL);
}
