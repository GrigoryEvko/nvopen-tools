// Function: sub_12D5E00
// Address: 0x12d5e00
//
__int64 __fastcall sub_12D5E00(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 *j; // rbx
  unsigned __int64 *v3; // r15
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *i; // r14
  __int64 v10; // rax
  void *v11; // [rsp+0h] [rbp-90h] BYREF
  char v12[16]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  void *v14; // [rsp+30h] [rbp-60h] BYREF
  char v15[16]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v16; // [rsp+48h] [rbp-48h]

  sub_1359CD0();
  if ( *(_DWORD *)(a1 + 48) )
  {
    sub_1359800(&v11, -8, 0);
    sub_1359800(&v14, -16, 0);
    v8 = *(_QWORD **)(a1 + 32);
    for ( i = &v8[6 * *(unsigned int *)(a1 + 48)]; i != v8; v8 += 6 )
    {
      v10 = v8[3];
      *v8 = &unk_49EE2B0;
      if ( v10 != 0 && v10 != -8 && v10 != -16 )
        sub_1649B30(v8 + 1);
    }
    v14 = &unk_49EE2B0;
    if ( v16 != 0 && v16 != -8 && v16 != -16 )
      sub_1649B30(v15);
    v11 = &unk_49EE2B0;
    if ( v13 != -8 && v13 != 0 && v13 != -16 )
      sub_1649B30(v12);
  }
  result = j___libc_free_0(*(_QWORD *)(a1 + 32));
  for ( j = *(unsigned __int64 **)(a1 + 16); (unsigned __int64 *)(a1 + 8) != j; result = j_j___libc_free_0(v3, 72) )
  {
    v3 = j;
    j = (unsigned __int64 *)j[1];
    v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *j = v4 | *j & 7;
    *(_QWORD *)(v4 + 8) = j;
    v5 = v3[6];
    v6 = v3[5];
    *v3 &= 7u;
    v3[1] = 0;
    if ( v5 != v6 )
    {
      do
      {
        v7 = *(_QWORD *)(v6 + 16);
        if ( v7 != -8 && v7 != 0 && v7 != -16 )
          sub_1649B30(v6);
        v6 += 24LL;
      }
      while ( v5 != v6 );
      v6 = v3[5];
    }
    if ( v6 )
      j_j___libc_free_0(v6, v3[7] - v6);
  }
  return result;
}
