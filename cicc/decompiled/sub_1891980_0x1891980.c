// Function: sub_1891980
// Address: 0x1891980
//
void *__fastcall sub_1891980(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-88h] BYREF
  __int64 v23; // [rsp+10h] [rbp-80h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+20h] [rbp-70h]
  __int64 (__fastcall **v26)(); // [rsp+30h] [rbp-60h]
  __int64 v27; // [rsp+38h] [rbp-58h] BYREF
  __int64 v28; // [rsp+40h] [rbp-50h]
  __int64 v29; // [rsp+48h] [rbp-48h]
  __int64 v30; // [rsp+50h] [rbp-40h]

  v1 = *(_BYTE *)(a1 + 384) == 0;
  *(_QWORD *)a1 = off_49F1CE8;
  if ( !v1 )
  {
    v14 = *(unsigned int *)(a1 + 376);
    if ( (_DWORD)v14 )
    {
      v15 = *(_QWORD **)(a1 + 360);
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
    j___libc_free_0(*(_QWORD *)(a1 + 360));
  }
  v2 = *(unsigned int *)(a1 + 344);
  if ( (_DWORD)v2 )
  {
    v8 = *(_QWORD **)(a1 + 328);
    v22 = 2;
    v23 = 0;
    v9 = &v8[6 * v2];
    v24 = -8;
    v25 = 0;
    v27 = 2;
    v28 = 0;
    v29 = -16;
    v26 = off_49F1D90;
    v30 = 0;
    do
    {
      v10 = v8[3];
      *v8 = &unk_49EE2B0;
      if ( v10 != 0 && v10 != -8 && v10 != -16 )
        sub_1649B30(v8 + 1);
      v8 += 6;
    }
    while ( v9 != v8 );
    v26 = (__int64 (__fastcall **)())&unk_49EE2B0;
    if ( v29 != -8 && v29 != 0 && v29 != -16 )
      sub_1649B30(&v27);
    if ( v24 != -8 && v24 != 0 && v24 != -16 )
      sub_1649B30(&v22);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 328));
  sub_1890C70(*(_QWORD *)(a1 + 288));
  v3 = *(_QWORD **)(a1 + 256);
  v4 = *(_QWORD **)(a1 + 248);
  if ( v3 != v4 )
  {
    do
    {
      v5 = v4[2];
      if ( v5 != -8 && v5 != 0 && v5 != -16 )
        sub_1649B30(v4);
      v4 += 3;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 248);
  }
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 264) - (_QWORD)v4);
  if ( *(_BYTE *)(a1 + 224) )
  {
    v18 = *(unsigned int *)(a1 + 216);
    if ( (_DWORD)v18 )
    {
      v19 = *(_QWORD **)(a1 + 200);
      v20 = &v19[2 * v18];
      do
      {
        if ( *v19 != -4 && *v19 != -8 )
        {
          v21 = v19[1];
          if ( v21 )
            sub_161E7C0((__int64)(v19 + 1), v21);
        }
        v19 += 2;
      }
      while ( v20 != v19 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 200));
  }
  v6 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v6 )
  {
    v11 = *(_QWORD **)(a1 + 168);
    v22 = 2;
    v23 = 0;
    v24 = -8;
    v12 = &v11[6 * v6];
    v25 = 0;
    v27 = 2;
    v28 = 0;
    v29 = -16;
    v26 = (__int64 (__fastcall **)())&unk_49F1DB8;
    v30 = 0;
    do
    {
      v13 = v11[3];
      *v11 = &unk_49EE2B0;
      if ( v13 != 0 && v13 != -8 && v13 != -16 )
        sub_1649B30(v11 + 1);
      v11 += 6;
    }
    while ( v12 != v11 );
    v26 = (__int64 (__fastcall **)())&unk_49EE2B0;
    if ( v29 != 0 && v29 != -8 && v29 != -16 )
      sub_1649B30(&v27);
    if ( v24 != 0 && v24 != -8 && v24 != -16 )
      sub_1649B30(&v22);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  return sub_1636790((_QWORD *)a1);
}
