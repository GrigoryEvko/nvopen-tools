// Function: sub_1C5A4D0
// Address: 0x1c5a4d0
//
void __fastcall sub_1C5A4D0(__int64 a1)
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
  _QWORD *v11; // r13
  _QWORD v12[3]; // [rsp+10h] [rbp-310h] BYREF
  char v13; // [rsp+28h] [rbp-2F8h]
  __int64 v14; // [rsp+30h] [rbp-2F0h] BYREF
  _QWORD *v15; // [rsp+38h] [rbp-2E8h]
  _QWORD *v16; // [rsp+40h] [rbp-2E0h]
  __int64 v17; // [rsp+48h] [rbp-2D8h]
  int v18; // [rsp+50h] [rbp-2D0h]
  _QWORD v19[8]; // [rsp+58h] [rbp-2C8h] BYREF
  __int64 v20; // [rsp+98h] [rbp-288h] BYREF
  __int64 v21; // [rsp+A0h] [rbp-280h]
  __int64 v22; // [rsp+A8h] [rbp-278h]
  _BYTE *v23; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v24; // [rsp+B8h] [rbp-268h]
  _BYTE v25[256]; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v26; // [rsp+1C0h] [rbp-160h] BYREF
  _BYTE *v27; // [rsp+1C8h] [rbp-158h]
  _BYTE *v28; // [rsp+1D0h] [rbp-150h]
  __int64 v29; // [rsp+1D8h] [rbp-148h]
  int v30; // [rsp+1E0h] [rbp-140h]
  _BYTE v31[312]; // [rsp+1E8h] [rbp-138h] BYREF

  v23 = v25;
  v24 = 0x2000000000LL;
  v27 = v31;
  v28 = v31;
  v1 = *(_QWORD *)(a1 + 80);
  v26 = 0;
  v29 = 32;
  if ( v1 )
    v1 -= 24;
  v30 = 0;
  v15 = v19;
  v16 = v19;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v17 = 0x100000008LL;
  v18 = 0;
  v19[0] = v1;
  v14 = 1;
  v12[0] = v1;
  v13 = 0;
  sub_144A690(&v20, (__int64)v12);
  while ( v20 != v21 )
  {
    sub_1C59C20(*(_QWORD *)(v21 - 32), (__int64)&v26, (__int64)&v23);
    sub_17D3A30((__int64)&v14);
  }
  if ( v20 )
    j_j___libc_free_0(v20, v22 - v20);
  if ( v16 != v15 )
    _libc_free((unsigned __int64)v16);
  v2 = (unsigned __int64)v23;
  v3 = *(_QWORD *)&v23[8 * (unsigned int)v24 - 8];
  v4 = (_DWORD)v24 == 1;
  v5 = (unsigned int)(v24 - 1);
  LODWORD(v24) = v24 - 1;
  if ( v4 )
  {
    v11 = (_QWORD *)v3;
  }
  else
  {
    while ( 1 )
    {
      v11 = *(_QWORD **)(v2 + 8 * v5 - 8);
      sub_1580AC0(v11, v3);
      v4 = (_DWORD)v24 == 1;
      v5 = (unsigned int)(v24 - 1);
      LODWORD(v24) = v24 - 1;
      if ( v4 )
        break;
      v2 = (unsigned __int64)v23;
      v3 = (__int64)v11;
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
      sub_1C59C20(v8, (__int64)&v26, (__int64)&v23);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v7 != v6 );
    v9 = v24;
    if ( (_DWORD)v24 )
    {
      do
      {
        v10 = (__int64)v11;
        v11 = *(_QWORD **)&v23[8 * v9 - 8];
        sub_1580AC0(v11, v10);
        v4 = (_DWORD)v24 == 1;
        v9 = v24 - 1;
        LODWORD(v24) = v24 - 1;
      }
      while ( !v4 );
    }
  }
  if ( v28 != v27 )
    _libc_free((unsigned __int64)v28);
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
}
