// Function: sub_197E0D0
// Address: 0x197e0d0
//
__int64 __fastcall sub_197E0D0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r13
  __int64 v18; // rsi
  _QWORD v19[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v20; // [rsp+18h] [rbp-78h]
  __int64 v21; // [rsp+20h] [rbp-70h]
  void *v22; // [rsp+30h] [rbp-60h]
  _QWORD v23[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int64 v25; // [rsp+50h] [rbp-40h]

  j___libc_free_0(*(_QWORD *)(a1 + 464));
  j___libc_free_0(*(_QWORD *)(a1 + 432));
  j___libc_free_0(*(_QWORD *)(a1 + 400));
  *(_QWORD *)(a1 + 176) = &unk_49EC708;
  v2 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 368);
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
  j___libc_free_0(*(_QWORD *)(a1 + 368));
  v6 = *(_QWORD *)(a1 + 216);
  if ( v6 != a1 + 232 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 96);
  if ( v7 != a1 + 112 )
    _libc_free(v7);
  if ( *(_BYTE *)(a1 + 80) )
  {
    v15 = *(unsigned int *)(a1 + 72);
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD **)(a1 + 56);
      v17 = &v16[2 * v15];
      do
      {
        if ( *v16 != -8 && *v16 != -4 )
        {
          v18 = v16[1];
          if ( v18 )
            sub_161E7C0((__int64)(v16 + 1), v18);
        }
        v16 += 2;
      }
      while ( v17 != v16 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 56));
  }
  v8 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v8 )
  {
    v10 = *(_QWORD **)(a1 + 24);
    v19[0] = 2;
    v19[1] = 0;
    v20 = -8;
    v11 = &v10[8 * v8];
    v22 = &unk_49E6B50;
    v12 = -8;
    v21 = 0;
    v23[0] = 2;
    v23[1] = 0;
    v24 = -16;
    v25 = 0;
    while ( 1 )
    {
      v13 = v10[3];
      if ( v13 != v12 )
      {
        v12 = v24;
        if ( v13 != v24 )
        {
          v14 = v10[7];
          if ( v14 != -8 && v14 != 0 && v14 != -16 )
          {
            sub_1649B30(v10 + 5);
            v13 = v10[3];
          }
          v12 = v13;
        }
      }
      *v10 = &unk_49EE2B0;
      if ( v12 != -8 && v12 != 0 && v12 != -16 )
        sub_1649B30(v10 + 1);
      v10 += 8;
      if ( v11 == v10 )
        break;
      v12 = v20;
    }
    v22 = &unk_49EE2B0;
    if ( v24 != 0 && v24 != -8 && v24 != -16 )
      sub_1649B30(v23);
    if ( v20 != -8 && v20 != 0 && v20 != -16 )
      sub_1649B30(v19);
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 24));
}
