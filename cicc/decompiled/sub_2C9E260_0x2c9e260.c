// Function: sub_2C9E260
// Address: 0x2c9e260
//
void __fastcall sub_2C9E260(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rsi
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // r13
  __m128i v12; // [rsp+10h] [rbp-300h] BYREF
  char v13; // [rsp+28h] [rbp-2E8h]
  __int64 v14; // [rsp+30h] [rbp-2E0h] BYREF
  __int64 *v15; // [rsp+38h] [rbp-2D8h]
  __int64 v16; // [rsp+40h] [rbp-2D0h]
  int v17; // [rsp+48h] [rbp-2C8h]
  char v18; // [rsp+4Ch] [rbp-2C4h]
  __int64 v19; // [rsp+50h] [rbp-2C0h] BYREF
  unsigned __int64 v20; // [rsp+90h] [rbp-280h] BYREF
  __int64 v21; // [rsp+98h] [rbp-278h]
  __int64 v22; // [rsp+A0h] [rbp-270h]
  _BYTE *v23; // [rsp+B0h] [rbp-260h] BYREF
  __int64 v24; // [rsp+B8h] [rbp-258h]
  _BYTE v25[256]; // [rsp+C0h] [rbp-250h] BYREF
  __int64 v26; // [rsp+1C0h] [rbp-150h] BYREF
  char *v27; // [rsp+1C8h] [rbp-148h]
  __int64 v28; // [rsp+1D0h] [rbp-140h]
  int v29; // [rsp+1D8h] [rbp-138h]
  char v30; // [rsp+1DCh] [rbp-134h]
  char v31; // [rsp+1E0h] [rbp-130h] BYREF

  v23 = v25;
  v24 = 0x2000000000LL;
  v27 = &v31;
  v1 = *(_QWORD *)(a1 + 80);
  v26 = 0;
  v28 = 32;
  if ( v1 )
    v1 -= 24;
  v29 = 0;
  v30 = 1;
  v15 = &v19;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v16 = 0x100000008LL;
  v17 = 0;
  v18 = 1;
  v19 = v1;
  v14 = 1;
  v12.m128i_i64[0] = v1;
  v13 = 0;
  sub_2C9A0C0((__int64)&v20, &v12);
  while ( v20 != v21 )
  {
    sub_2C9D870(*(_QWORD *)(v21 - 32), (__int64)&v26, (__int64)&v23);
    sub_23EC7E0((__int64)&v14);
  }
  if ( v20 )
    j_j___libc_free_0(v20);
  if ( !v18 )
    _libc_free((unsigned __int64)v15);
  v2 = (unsigned __int64)v23;
  v3 = *(_QWORD *)&v23[8 * (unsigned int)v24 - 8];
  v4 = (_DWORD)v24 == 1;
  v5 = (unsigned int)(v24 - 1);
  LODWORD(v24) = v24 - 1;
  if ( v4 )
  {
    v11 = v3;
  }
  else
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v2 + 8 * v5 - 8);
      sub_AA4AF0(v11, v3);
      v4 = (_DWORD)v24 == 1;
      v5 = (unsigned int)(v24 - 1);
      LODWORD(v24) = v24 - 1;
      if ( v4 )
        break;
      v2 = (unsigned __int64)v23;
      v3 = v11;
    }
  }
  v6 = *(_QWORD *)(a1 + 80);
  v7 = a1 + 72;
  if ( a1 + 72 != v6 )
  {
    do
    {
      v8 = v6 - 24;
      if ( !v6 )
        v8 = 0;
      sub_2C9D870(v8, (__int64)&v26, (__int64)&v23);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v7 != v6 );
    v9 = v24;
    if ( (_DWORD)v24 )
    {
      do
      {
        v10 = v11;
        v11 = *(_QWORD *)&v23[8 * v9 - 8];
        sub_AA4AF0(v11, v10);
        v4 = (_DWORD)v24 == 1;
        v9 = v24 - 1;
        LODWORD(v24) = v24 - 1;
      }
      while ( !v4 );
    }
  }
  if ( !v30 )
    _libc_free((unsigned __int64)v27);
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
}
