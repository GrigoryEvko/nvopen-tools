// Function: sub_3113320
// Address: 0x3113320
//
__int64 __fastcall sub_3113320(__int64 a1, void **a2, __int64 *a3)
{
  unsigned __int64 v4; // r13
  __int64 v6; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-60h] BYREF
  char v8; // [rsp+10h] [rbp-50h]
  __int64 v9[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v10; // [rsp+30h] [rbp-30h] BYREF

  sub_CA0F50(v9, a2);
  if ( sub_2241AC0((__int64)v9, "-") )
    sub_CA4130((__int64)v7, a3, (__int64)a2, -1, 1u, 0, 1);
  else
    sub_C7DF90((__int64)v7);
  if ( (__int64 *)v9[0] != &v10 )
    j_j___libc_free_0(v9[0]);
  if ( (v8 & 1) == 0 || !LODWORD(v7[0]) )
  {
    v6 = v7[0];
    goto LABEL_12;
  }
  sub_C63CA0(v9, v7[0], v7[1]);
  v4 = v9[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v8 & 1) == 0 && v7[0] )
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v7[0] + 8LL))(v7[0]);
    if ( !v4 )
      goto LABEL_17;
LABEL_9:
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v4;
    return a1;
  }
  if ( v4 )
    goto LABEL_9;
LABEL_17:
  v6 = 0;
LABEL_12:
  v9[0] = v6;
  sub_3112F20(a1, v9);
  if ( v9[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9[0] + 8LL))(v9[0]);
  return a1;
}
