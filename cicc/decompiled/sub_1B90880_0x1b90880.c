// Function: sub_1B90880
// Address: 0x1b90880
//
void __fastcall sub_1B90880(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // rdx
  _QWORD *v14; // rbx
  _QWORD *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r13
  __int64 v21; // rsi
  _QWORD v22[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  void *v25; // [rsp+30h] [rbp-50h]
  _QWORD v26[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v27; // [rsp+48h] [rbp-38h]
  __int64 v28; // [rsp+50h] [rbp-30h]

  *(_QWORD *)a1 = &unk_49F6E20;
  j___libc_free_0(*(_QWORD *)(a1 + 480));
  v2 = *(_QWORD *)(a1 + 384);
  if ( v2 != a1 + 400 )
    _libc_free(v2);
  sub_1B8EA10(*(_QWORD *)(a1 + 352));
  sub_1B8F140(*(_QWORD **)(a1 + 304));
  v3 = *(_QWORD *)(a1 + 216);
  if ( v3 != a1 + 232 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 96);
  if ( v4 )
    sub_161E7C0(a1 + 96, v4);
  v5 = *(_QWORD *)(a1 + 80);
  if ( v5 )
  {
    j___libc_free_0(*(_QWORD *)(v5 + 464));
    j___libc_free_0(*(_QWORD *)(v5 + 432));
    j___libc_free_0(*(_QWORD *)(v5 + 400));
    *(_QWORD *)(v5 + 176) = &unk_49EC708;
    v6 = *(unsigned int *)(v5 + 384);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD **)(v5 + 368);
      v8 = &v7[7 * v6];
      do
      {
        if ( *v7 != -16 && *v7 != -8 )
        {
          v9 = v7[1];
          if ( (_QWORD *)v9 != v7 + 3 )
            _libc_free(v9);
        }
        v7 += 7;
      }
      while ( v8 != v7 );
    }
    j___libc_free_0(*(_QWORD *)(v5 + 368));
    v10 = *(_QWORD *)(v5 + 216);
    if ( v10 != v5 + 232 )
      _libc_free(v10);
    v11 = *(_QWORD *)(v5 + 96);
    if ( v11 != v5 + 112 )
      _libc_free(v11);
    if ( *(_BYTE *)(v5 + 80) )
    {
      v18 = *(unsigned int *)(v5 + 72);
      if ( (_DWORD)v18 )
      {
        v19 = *(_QWORD **)(v5 + 56);
        v20 = &v19[2 * v18];
        do
        {
          if ( *v19 != -8 && *v19 != -4 )
          {
            v21 = v19[1];
            if ( v21 )
              sub_161E7C0((__int64)(v19 + 1), v21);
          }
          v19 += 2;
        }
        while ( v20 != v19 );
      }
      j___libc_free_0(*(_QWORD *)(v5 + 56));
      v12 = *(unsigned int *)(v5 + 40);
      if ( !(_DWORD)v12 )
        goto LABEL_21;
    }
    else
    {
      v12 = *(unsigned int *)(v5 + 40);
      if ( !(_DWORD)v12 )
      {
LABEL_21:
        j___libc_free_0(*(_QWORD *)(v5 + 24));
        j_j___libc_free_0(v5, 520);
        return;
      }
    }
    v23 = -8;
    v13 = -8;
    v24 = 0;
    v27 = -16;
    v25 = &unk_49E6B50;
    v28 = 0;
    v14 = *(_QWORD **)(v5 + 24);
    v22[0] = 2;
    v22[1] = 0;
    v15 = &v14[8 * v12];
    v26[0] = 2;
    v26[1] = 0;
    while ( 1 )
    {
      v16 = v14[3];
      if ( v13 != v16 && v16 != v27 )
        sub_1455FA0((__int64)(v14 + 5));
      *v14 = &unk_49EE2B0;
      v17 = (__int64)(v14 + 1);
      v14 += 8;
      sub_1455FA0(v17);
      if ( v15 == v14 )
        break;
      v13 = v23;
    }
    v25 = &unk_49EE2B0;
    sub_1455FA0((__int64)v26);
    sub_1455FA0((__int64)v22);
    goto LABEL_21;
  }
}
