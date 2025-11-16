// Function: sub_31D3FC0
// Address: 0x31d3fc0
//
__int64 __fastcall sub_31D3FC0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r12d
  _QWORD *v5; // rax
  unsigned __int64 v6; // rax
  bool v7; // cc
  unsigned int v9; // eax
  unsigned __int64 v10; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-98h]
  unsigned __int64 v12; // [rsp+10h] [rbp-90h]
  unsigned int v13; // [rsp+18h] [rbp-88h]
  unsigned __int64 v14; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-78h]
  unsigned __int64 v16; // [rsp+30h] [rbp-70h]
  unsigned int v17; // [rsp+38h] [rbp-68h]
  unsigned __int64 v18; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-58h]
  unsigned __int64 v20; // [rsp+50h] [rbp-50h]
  unsigned int v21; // [rsp+58h] [rbp-48h]
  char v22; // [rsp+60h] [rbp-40h]

  v4 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8;
  v19 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43690((__int64)&v18, a2, 0);
    v15 = v4;
    sub_C43690((__int64)&v14, a1, 0);
  }
  else
  {
    v18 = a2;
    v15 = v4;
    v14 = a1;
  }
  sub_AADC30((__int64)&v10, (__int64)&v14, (__int64 *)&v18);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  sub_B492D0((__int64)&v18, a3);
  if ( v22 )
  {
    sub_AB2160((__int64)&v14, (__int64)&v10, (__int64)&v18, 0);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    v10 = v14;
    v9 = v15;
    v15 = 0;
    v11 = v9;
    if ( v13 > 0x40 && v12 )
    {
      j_j___libc_free_0_0(v12);
      v12 = v16;
      v13 = v17;
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
    }
    else
    {
      v12 = v16;
      v13 = v17;
    }
    if ( v22 )
    {
      v22 = 0;
      if ( v21 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
      if ( v19 > 0x40 && v18 )
        j_j___libc_free_0_0(v18);
    }
  }
  v5 = (_QWORD *)sub_BD5C60(a3);
  v6 = sub_A7B4D0((__int64 *)(a3 + 72), v5, (__int64)&v10);
  v7 = v13 <= 0x40;
  *(_QWORD *)(a3 + 72) = v6;
  if ( !v7 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return 1;
}
