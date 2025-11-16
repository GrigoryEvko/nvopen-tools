// Function: sub_A1BA60
// Address: 0xa1ba60
//
__int64 __fastcall sub_A1BA60(__int64 a1)
{
  __int64 *v1; // r13
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  _QWORD *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 *v9; // rcx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // [rsp+0h] [rbp-F0h]
  __int64 *v16; // [rsp+8h] [rbp-E8h]
  _QWORD *v17; // [rsp+18h] [rbp-D8h]
  __int64 v18; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v19; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v20; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v22; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v23; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v24; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+58h] [rbp-98h] BYREF
  unsigned __int64 v26[2]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD v27[2]; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v28; // [rsp+80h] [rbp-70h] BYREF
  __int64 v29; // [rsp+88h] [rbp-68h]
  __int64 v30; // [rsp+90h] [rbp-60h] BYREF
  char v31; // [rsp+98h] [rbp-58h] BYREF
  __int64 v32; // [rsp+A0h] [rbp-50h]
  __int64 v33; // [rsp+A8h] [rbp-48h]
  __int64 v34; // [rsp+B0h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 168);
  if ( *(__int64 **)(a1 + 160) == v1 )
  {
    v5 = *(_QWORD *)(a1 + 168);
LABEL_12:
    v29 = 0;
    v28 = (__int64 *)&v31;
    *(_BYTE *)(a1 + 153) = 1;
    v30 = 0;
    sub_C0F6D0(&v18, v1, (v5 - (__int64)v1) >> 3, &v28, a1 + 8, a1 + 56);
    v6 = (_QWORD *)(v18 & 0xFFFFFFFFFFFFFFFELL);
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v18 = 0;
      v7 = *v6;
      v19 = 0;
      v8 = (__int64)&unk_4F84052;
      v20 = 0;
      v21 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(v7 + 48))(v6, &unk_4F84052) )
      {
        v9 = (__int64 *)v6[2];
        v10 = (__int64 *)v6[1];
        v22 = 1;
        v15 = v9;
        if ( v10 == v9 )
        {
          v13 = 1;
        }
        else
        {
          do
          {
            v16 = v10;
            v25 = *v10;
            *v10 = 0;
            sub_A16FF0(&v24, &v25);
            v11 = v22;
            v8 = (__int64)v26;
            v22 = 0;
            v26[0] = v11 | 1;
            sub_9CDB40(&v23, v26, (unsigned __int64 *)&v24);
            if ( (v22 & 1) != 0 || (v12 = v16, (v22 & 0xFFFFFFFFFFFFFFFELL) != 0) )
              sub_C63C30(&v22);
            v22 |= v23 | 1;
            if ( (v26[0] & 1) != 0 || (v26[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(v26);
            if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v24);
            if ( v25 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
              v12 = v16;
            }
            v10 = v12 + 1;
          }
          while ( v15 != v10 );
          v13 = v22 | 1;
        }
        v25 = v13;
        (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
      }
      else
      {
        v8 = (__int64)v26;
        v26[0] = (unsigned __int64)v6;
        sub_A16FF0(&v25, v26);
        if ( v26[0] )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v26[0] + 8LL))(v26[0]);
      }
      if ( (v25 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      v25 = 0;
      sub_9C66B0(&v25);
      sub_9C66B0(&v21);
      sub_9C66B0(&v20);
      sub_9C66B0(&v19);
      result = sub_9C66B0(&v18);
      v14 = (__int64)v28;
      if ( v28 == (__int64 *)&v31 )
        return result;
    }
    else
    {
      v8 = 25;
      result = sub_A1B8B0((__int64 *)a1, 0x19u, 1u, (__int64)v28, v29);
      v14 = (__int64)v28;
      if ( v28 == (__int64 *)&v31 )
        return result;
    }
    return _libc_free(v14, v8);
  }
  v2 = *(__int64 **)(a1 + 160);
  while ( 1 )
  {
    v3 = *v2;
    if ( *(_QWORD *)(*v2 + 96) )
      break;
LABEL_10:
    if ( v1 == ++v2 )
    {
      v1 = *(__int64 **)(a1 + 160);
      v5 = *(_QWORD *)(a1 + 168);
      goto LABEL_12;
    }
  }
  LOBYTE(v27[0]) = 0;
  v26[0] = (unsigned __int64)v27;
  v26[1] = 0;
  v28 = &v30;
  v17 = (_QWORD *)v3;
  sub_A15D40((__int64 *)&v28, *(_BYTE **)(v3 + 232), *(_QWORD *)(v3 + 232) + *(_QWORD *)(v3 + 240));
  v32 = v17[33];
  v33 = v17[34];
  v34 = v17[35];
  result = sub_C0D4F0(&v28, v26);
  if ( result && *(_QWORD *)(result + 112) )
  {
    if ( v28 != &v30 )
      j_j___libc_free_0(v28, v30 + 1);
    if ( (_QWORD *)v26[0] != v27 )
      j_j___libc_free_0(v26[0], v27[0] + 1LL);
    goto LABEL_10;
  }
  if ( v28 != &v30 )
    result = j_j___libc_free_0(v28, v30 + 1);
  if ( (_QWORD *)v26[0] != v27 )
    return j_j___libc_free_0(v26[0], v27[0] + 1LL);
  return result;
}
