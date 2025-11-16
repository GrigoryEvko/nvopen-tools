// Function: sub_1B93DB0
// Address: 0x1b93db0
//
__int64 __fastcall sub_1B93DB0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r13
  __int64 v15; // rsi
  _QWORD v16[5]; // [rsp+8h] [rbp-88h] BYREF
  void *v17; // [rsp+30h] [rbp-60h]
  _QWORD v18[11]; // [rsp+38h] [rbp-58h] BYREF

  *(_QWORD *)(a1 + 128) = &unk_49EC708;
  v2 = *(unsigned int *)(a1 + 336);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 320);
    v4 = &v3[7 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 7;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 320));
  v6 = *(_QWORD *)(a1 + 168);
  if ( v6 != a1 + 184 )
    _libc_free(v6);
  if ( *(_BYTE *)(a1 + 96) )
  {
    v12 = *(unsigned int *)(a1 + 88);
    if ( (_DWORD)v12 )
    {
      v13 = *(_QWORD **)(a1 + 72);
      v14 = &v13[2 * v12];
      do
      {
        if ( *v13 != -8 && *v13 != -4 )
        {
          v15 = v13[1];
          if ( v15 )
            sub_161E7C0((__int64)(v13 + 1), v15);
        }
        v13 += 2;
      }
      while ( v14 != v13 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 72));
  }
  v7 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v7 )
  {
    v9 = *(_QWORD **)(a1 + 40);
    v16[0] = 2;
    v16[1] = 0;
    v16[2] = -8;
    v10 = &v9[6 * v7];
    v16[3] = 0;
    v18[0] = 2;
    v18[1] = 0;
    v18[2] = -16;
    v17 = &unk_49EC740;
    v18[3] = 0;
    do
    {
      v11 = v9[3];
      *v9 = &unk_49EE2B0;
      if ( v11 != -8 && v11 != 0 && v11 != -16 )
        sub_1649B30(v9 + 1);
      v9 += 6;
    }
    while ( v10 != v9 );
    v17 = &unk_49EE2B0;
    sub_1455FA0((__int64)v18);
    sub_1455FA0((__int64)v16);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 40));
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
