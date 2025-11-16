// Function: sub_31F8400
// Address: 0x31f8400
//
__int64 __fastcall sub_31F8400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ecx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  size_t v10; // r12
  char *v11; // rax
  __int64 v12; // rax
  unsigned int v13; // r12d
  void *v15; // [rsp+0h] [rbp-70h] BYREF
  size_t v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  _BYTE s[8]; // [rsp+18h] [rbp-58h] BYREF
  __int16 v19; // [rsp+20h] [rbp-50h] BYREF
  char *v20; // [rsp+28h] [rbp-48h]
  size_t v21; // [rsp+30h] [rbp-40h]
  unsigned __int64 v22; // [rsp+38h] [rbp-38h]
  __int64 v23; // [rsp+40h] [rbp-30h]
  __int64 v24; // [rsp+48h] [rbp-28h]

  v6 = *(_QWORD *)(a1 + 8);
  v15 = s;
  v7 = *(_DWORD *)(*(_QWORD *)(v6 + 208) + 8LL);
  v8 = *(_QWORD *)(a2 + 24);
  v16 = 0;
  v17 = 4;
  v9 = v8 / (unsigned int)(8 * v7);
  v10 = (unsigned int)v9;
  if ( (unsigned int)v9 > 4 )
  {
    sub_C8D290((__int64)&v15, s, (unsigned int)v9, 1u, a5, a6);
    v11 = (char *)v15 + v10;
    if ( v15 != (char *)v15 + v10 )
    {
      memset(v15, 5, v10);
      v11 = (char *)v15;
    }
    v16 = v10;
  }
  else
  {
    if ( (_DWORD)v9 )
      memset(s, 5, (unsigned int)v9);
    v16 = v10;
    v11 = s;
  }
  v21 = v10;
  v19 = 10;
  v20 = v11;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v12 = sub_370A6C0(a1 + 648, &v19);
  v13 = sub_3707F80(a1 + 632, v12);
  if ( v22 )
    j_j___libc_free_0(v22);
  if ( v15 != s )
    _libc_free((unsigned __int64)v15);
  return v13;
}
