// Function: sub_1B90BF0
// Address: 0x1b90bf0
//
__int64 __fastcall sub_1B90BF0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rsi
  _QWORD v18[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h]
  __int64 v20; // [rsp+20h] [rbp-70h]
  void *v21; // [rsp+30h] [rbp-60h]
  _QWORD v22[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v23; // [rsp+48h] [rbp-48h]
  __int64 v24; // [rsp+50h] [rbp-40h]

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
    v14 = *(unsigned int *)(a1 + 72);
    if ( (_DWORD)v14 )
    {
      v15 = *(_QWORD **)(a1 + 56);
      v16 = &v15[2 * v14];
      do
      {
        if ( *v15 != -8 && *v15 != -4 )
        {
          v17 = v15[1];
          if ( v17 )
            sub_161E7C0((__int64)(v15 + 1), v17);
        }
        v15 += 2;
      }
      while ( v16 != v15 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 56));
  }
  v8 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v8 )
  {
    v10 = *(_QWORD **)(a1 + 24);
    v18[0] = 2;
    v18[1] = 0;
    v19 = -8;
    v11 = &v10[8 * v8];
    v21 = &unk_49E6B50;
    v12 = -8;
    v20 = 0;
    v22[0] = 2;
    v22[1] = 0;
    v23 = -16;
    v24 = 0;
    while ( 1 )
    {
      v13 = v10[3];
      if ( v13 != v12 )
      {
        v12 = v23;
        if ( v13 != v23 )
        {
          sub_1455FA0((__int64)(v10 + 5));
          v12 = v10[3];
        }
      }
      *v10 = &unk_49EE2B0;
      if ( v12 != -8 && v12 != 0 && v12 != -16 )
        sub_1649B30(v10 + 1);
      v10 += 8;
      if ( v11 == v10 )
        break;
      v12 = v19;
    }
    v21 = &unk_49EE2B0;
    sub_1455FA0((__int64)v22);
    sub_1455FA0((__int64)v18);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 24));
  return j_j___libc_free_0(a1, 520);
}
