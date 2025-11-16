// Function: sub_C64870
// Address: 0xc64870
//
__int64 __fastcall sub_C64870(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  _BYTE *v5; // r13
  __int64 v6; // rsi
  _BYTE *v7; // r14
  _BYTE *v8; // rax
  __int64 v9; // rsi
  _BYTE *v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rcx
  _QWORD *v14; // r13
  _QWORD *v16; // r14
  __int64 *v17; // rax
  __int64 *v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rax
  __int64 *v22; // [rsp+8h] [rbp-E8h]
  __int64 *v23; // [rsp+28h] [rbp-C8h]
  _QWORD *v24; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v26; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v28; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v29; // [rsp+58h] [rbp-98h] BYREF
  __int64 v30; // [rsp+60h] [rbp-90h] BYREF
  __int64 v31; // [rsp+68h] [rbp-88h] BYREF
  _BYTE *v32; // [rsp+70h] [rbp-80h] BYREF
  __int64 v33; // [rsp+78h] [rbp-78h]
  _BYTE v34[112]; // [rsp+80h] [rbp-70h] BYREF

  v24 = &v32;
  v2 = *a2;
  v32 = v34;
  v33 = 0x200000000LL;
  *a2 = 0;
  v25 = 0;
  v3 = v2 & 0xFFFFFFFFFFFFFFFELL;
  if ( v3 )
  {
    v16 = (_QWORD *)v3;
    a2 = (__int64 *)&unk_4F84052;
    v26 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v3 + 48LL))(v3, &unk_4F84052) )
    {
      v17 = (__int64 *)v16[2];
      v18 = (__int64 *)v16[1];
      v27 = 1;
      v22 = v17;
      if ( v18 == v17 )
      {
        v21 = 1;
      }
      else
      {
        do
        {
          v23 = v18;
          v29 = *v18;
          *v18 = 0;
          sub_C64610(&v30, &v29, (__int64 *)&v24);
          v19 = v27;
          a2 = &v28;
          v27 = 0;
          v28 = v19 | 1;
          sub_9CDB40((unsigned __int64 *)&v31, (unsigned __int64 *)&v28, (unsigned __int64 *)&v30);
          if ( (v27 & 1) != 0 || (v20 = v23, (v27 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v27, (__int64)&v28);
          v27 |= v31 | 1;
          if ( (v28 & 1) != 0 || (v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v28, (__int64)&v28);
          if ( (v30 & 1) != 0 || (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v30, (__int64)&v28);
          if ( v29 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
            v20 = v23;
          }
          v18 = v20 + 1;
        }
        while ( v22 != v18 );
        v21 = v27 | 1;
      }
      v30 = v21;
      (*(void (__fastcall **)(_QWORD *))(*v16 + 8LL))(v16);
    }
    else
    {
      v31 = (__int64)v16;
      a2 = &v31;
      sub_C64610(&v30, &v31, (__int64 *)&v24);
      if ( v31 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
    }
    if ( (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v26 & 1) != 0 || (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v26, (__int64)a2);
  }
  if ( (v25 & 1) != 0 || (v25 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v25, (__int64)a2);
  v4 = (unsigned int)v33;
  v5 = v32;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v6 = 32 * v4;
  *(_BYTE *)(a1 + 16) = 0;
  v7 = &v5[v6];
  if ( v5 != &v5[v6] )
  {
    v8 = v5;
    v9 = (v6 >> 5) - 1;
    do
    {
      v9 += *((_QWORD *)v8 + 1);
      v8 += 32;
    }
    while ( v7 != v8 );
    v10 = v5 + 32;
    sub_2240E30(a1, v9);
    v6 = *((_QWORD *)v10 - 4);
    sub_2241490(a1, v6, *((_QWORD *)v10 - 3), v11);
    while ( v7 != v10 )
    {
      if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v10 += 32;
      sub_2241490(a1, "\n", 1, v12);
      v6 = *((_QWORD *)v10 - 4);
      sub_2241490(a1, v6, *((_QWORD *)v10 - 3), v13);
    }
    v7 = v32;
    v14 = &v32[32 * (unsigned int)v33];
    if ( v14 != (_QWORD *)v32 )
    {
      do
      {
        v14 -= 4;
        if ( (_QWORD *)*v14 != v14 + 2 )
        {
          v6 = v14[2] + 1LL;
          j_j___libc_free_0(*v14, v6);
        }
      }
      while ( v14 != (_QWORD *)v7 );
      v7 = v32;
    }
  }
  if ( v7 != v34 )
    _libc_free(v7, v6);
  return a1;
}
