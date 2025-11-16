// Function: sub_26661E0
// Address: 0x26661e0
//
__int64 __fastcall sub_26661E0(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // rax
  _QWORD *v10; // r12
  _QWORD *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r13
  __int64 v16; // rsi
  void *v17; // [rsp+0h] [rbp-90h] BYREF
  __int64 v18; // [rsp+8h] [rbp-88h] BYREF
  __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+18h] [rbp-78h]
  __int64 v21; // [rsp+20h] [rbp-70h]
  void *v22; // [rsp+30h] [rbp-60h]
  __int64 v23; // [rsp+38h] [rbp-58h] BYREF
  __int64 v24; // [rsp+40h] [rbp-50h]
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  sub_C7D6A0(*(_QWORD *)(a1 + 264), 16LL * *(unsigned int *)(a1 + 280), 8);
  v1 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v1 )
  {
    v7 = *(_QWORD **)(a1 + 232);
    v19 = -4096;
    v17 = 0;
    v18 = 0;
    v8 = &v7[4 * v1];
    v22 = 0;
    v23 = 0;
    v24 = -8192;
    do
    {
      v9 = v7[2];
      if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        sub_BD60C0(v7);
      v7 += 4;
    }
    while ( v8 != v7 );
    if ( v19 != -4096 && v19 != 0 )
      sub_BD60C0(&v17);
    v1 = *(unsigned int *)(a1 + 248);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 32 * v1, 8);
  sub_2665870(*(_QWORD **)(a1 + 192));
  if ( !*(_BYTE *)(a1 + 140) )
    _libc_free(*(_QWORD *)(a1 + 120));
  v2 = *(_QWORD **)(a1 + 96);
  v3 = *(_QWORD **)(a1 + 88);
  if ( v2 != v3 )
  {
    do
    {
      v4 = v3[2];
      if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
        sub_BD60C0(v3);
      v3 += 3;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)(a1 + 88);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  if ( *(_BYTE *)(a1 + 64) )
  {
    v13 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 64) = 0;
    if ( (_DWORD)v13 )
    {
      v14 = *(_QWORD **)(a1 + 40);
      v15 = &v14[2 * v13];
      do
      {
        if ( *v14 != -4096 && *v14 != -8192 )
        {
          v16 = v14[1];
          if ( v16 )
            sub_B91220((__int64)(v14 + 1), v16);
        }
        v14 += 2;
      }
      while ( v15 != v14 );
      LODWORD(v13) = *(_DWORD *)(a1 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * (unsigned int)v13, 8);
  }
  v5 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v5 )
  {
    v10 = *(_QWORD **)(a1 + 8);
    v18 = 2;
    v19 = 0;
    v20 = -4096;
    v11 = &v10[6 * v5];
    v17 = &unk_4A1FB50;
    v21 = 0;
    v23 = 2;
    v24 = 0;
    v25 = -8192;
    v22 = &unk_4A1FB50;
    v26 = 0;
    do
    {
      v12 = v10[3];
      *v10 = &unk_49DB368;
      if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
        sub_BD60C0(v10 + 1);
      v10 += 6;
    }
    while ( v11 != v10 );
    v22 = &unk_49DB368;
    if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
      sub_BD60C0(&v23);
    v17 = &unk_49DB368;
    if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
      sub_BD60C0(&v18);
    v5 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 48 * v5, 8);
}
