// Function: sub_16B1F90
// Address: 0x16b1f90
//
__int64 __fastcall sub_16B1F90(__int64 a1, __int64 a2, const char *a3, size_t a4, __int64 a5, __int64 a6)
{
  const char *v6; // r15
  size_t v7; // r12
  _QWORD *v8; // r12
  _BYTE *v9; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  _QWORD *v16; // [rsp+0h] [rbp-80h] BYREF
  __int64 v17; // [rsp+8h] [rbp-78h]
  _QWORD v18[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v19[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int64 v21; // [rsp+38h] [rbp-48h]
  int v22; // [rsp+40h] [rbp-40h]
  _QWORD *v23; // [rsp+48h] [rbp-38h]

  if ( a3 )
  {
    v6 = a3;
    v7 = a4;
  }
  else
  {
    v6 = *(const char **)(a1 + 24);
    v7 = *(_QWORD *)(a1 + 32);
  }
  v17 = 0;
  v16 = v18;
  LOBYTE(v18[0]) = 0;
  v22 = 1;
  v21 = 0;
  v20 = 0;
  v19[1] = 0;
  v19[0] = &unk_49EFBE0;
  v23 = &v16;
  if ( v7 )
  {
    v11 = sub_16B0440();
    v12 = sub_16E7EE0(v19, *(const char **)v11, *(_QWORD *)(v11 + 8));
    v13 = sub_1263B40(v12, ": for the -");
    sub_1549FF0(v13, v6, v7);
    v14 = v21;
    v15 = v20 - v21;
  }
  else
  {
    if ( !*(_QWORD *)(a1 + 48) )
    {
LABEL_5:
      v8 = (_QWORD *)sub_16E7EE0(v19, " option: ", 9, a4, a5, a6, v16, v17, v18[0]);
      goto LABEL_6;
    }
    sub_16E7EE0(v19, *(const char **)(a1 + 40));
    v14 = v21;
    v15 = v20 - v21;
  }
  if ( v15 <= 8 )
    goto LABEL_5;
  *(_BYTE *)(v14 + 8) = 32;
  v8 = v19;
  *(_QWORD *)v14 = 0x3A6E6F6974706F20LL;
  v21 += 9;
LABEL_6:
  sub_16E2CE0(a2, v8);
  v9 = (_BYTE *)v8[3];
  if ( (_BYTE *)v8[2] == v9 )
  {
    sub_16E7EE0(v8, "\n", 1);
  }
  else
  {
    *v9 = 10;
    ++v8[3];
  }
  sub_16E7BC0(v19);
  if ( v16 != v18 )
    j_j___libc_free_0(v16, v18[0] + 1LL);
  return 1;
}
