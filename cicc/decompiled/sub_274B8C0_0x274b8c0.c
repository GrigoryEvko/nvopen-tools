// Function: sub_274B8C0
// Address: 0x274b8c0
//
__int64 __fastcall sub_274B8C0(unsigned __int8 *a1, __int64 *a2)
{
  _BOOL4 v2; // ebx
  bool v3; // al
  unsigned int v4; // r13d
  __int64 *v6; // r9
  unsigned __int8 *v7; // rdx
  char v8; // r12
  int v9; // eax
  char v10; // bl
  char v11; // dl
  bool v12; // [rsp+10h] [rbp-A0h]
  unsigned int v13; // [rsp+1Ch] [rbp-94h]
  unsigned __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-88h]
  unsigned __int64 v16; // [rsp+30h] [rbp-80h]
  unsigned int v17; // [rsp+38h] [rbp-78h]
  unsigned __int64 v18; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-68h]
  unsigned __int64 v20; // [rsp+50h] [rbp-60h]
  unsigned int v21; // [rsp+58h] [rbp-58h]
  unsigned __int64 v22; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-48h]
  unsigned __int64 v24; // [rsp+70h] [rbp-40h]
  unsigned int v25; // [rsp+78h] [rbp-38h]

  v2 = sub_B44900((__int64)a1);
  v3 = sub_B448F0((__int64)a1);
  v4 = v2;
  LOBYTE(v4) = v3 && v2;
  if ( v3 && v2 )
  {
    return 0;
  }
  else
  {
    v13 = *a1 - 29;
    if ( (a1[7] & 0x40) != 0 )
      v6 = (__int64 *)*((_QWORD *)a1 - 1);
    else
      v6 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v12 = v3;
    sub_22CEA30((__int64)&v14, a2, v6, 0);
    if ( (a1[7] & 0x40) != 0 )
      v7 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
    else
      v7 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v8 = 0;
    sub_22CEA30((__int64)&v18, a2, (__int64 *)v7 + 4, 0);
    if ( v12 )
      goto LABEL_9;
    sub_AB28E0((__int64)&v22, v13, (__int64)&v18, 1);
    v4 = sub_AB1BB0((__int64)&v22, (__int64)&v14);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    v8 = v4;
    if ( !v2 )
    {
LABEL_9:
      sub_AB28E0((__int64)&v22, v13, (__int64)&v18, 2);
      v9 = sub_AB1BB0((__int64)&v22, (__int64)&v14);
      v10 = v9;
      v4 |= v9;
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
      v11 = v10;
    }
    else
    {
      v11 = 0;
    }
    sub_274B030(a1, v13, v11, v8);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  return v4;
}
