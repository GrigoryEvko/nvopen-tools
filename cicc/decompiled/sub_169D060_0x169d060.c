// Function: sub_169D060
// Address: 0x169d060
//
_QWORD *__fastcall sub_169D060(_QWORD *a1, __int64 a2, __int64 *a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rax
  void *v9; // rax
  __int64 v10; // rdi
  void *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdi
  _QWORD *v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  *a1 = a2;
  v5 = (_QWORD *)sub_2207820(72);
  if ( v5 )
  {
    *v5 = 2;
    v6 = v5;
    v15 = v5 + 1;
    v7 = a3;
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      v7 = (__int64 *)*a3;
    v8 = *v7;
    v17 = 64;
    v16 = v8;
    v9 = sub_16982C0();
    v10 = (__int64)(v6 + 2);
    v11 = v9;
    if ( v9 == &unk_42AE9D0 )
      sub_169D060(v10, v9, &v16);
    else
      sub_169D050(v10, &unk_42AE9D0, &v16);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      a3 = (__int64 *)*a3;
    v12 = a3[1];
    v13 = (__int64)(v6 + 6);
    v17 = 64;
    v16 = v12;
    if ( v11 == &unk_42AE9D0 )
      sub_169D060(v13, v11, &v16);
    else
      sub_169D050(v13, &unk_42AE9D0, &v16);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
  }
  else
  {
    v15 = 0;
  }
  a1[1] = v15;
  return v15;
}
