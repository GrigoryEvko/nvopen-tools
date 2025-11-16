// Function: sub_15E1360
// Address: 0x15e1360
//
__int64 __fastcall sub_15E1360(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  _BYTE *v10; // rsi
  __int64 v11; // r12
  __int64 v13; // [rsp+8h] [rbp-F8h]
  __int64 v14; // [rsp+18h] [rbp-E8h]
  unsigned int *v15; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-D8h]
  unsigned int *v17; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v18; // [rsp+38h] [rbp-C8h]
  _BYTE v19[64]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE *v20; // [rsp+80h] [rbp-80h] BYREF
  __int64 v21; // [rsp+88h] [rbp-78h]
  _BYTE v22[112]; // [rsp+90h] [rbp-70h] BYREF

  v18 = 0x800000000LL;
  v17 = (unsigned int *)v19;
  sub_15E1220(a2, (__int64)&v17);
  v15 = v17;
  v16 = (unsigned int)v18;
  v4 = sub_15DE7A0(&v15, a3, a1);
  v5 = v16;
  v20 = v22;
  v14 = v4;
  v21 = 0x800000000LL;
  if ( !v16 )
  {
    v10 = v22;
    goto LABEL_7;
  }
  do
  {
    v6 = sub_15DE7A0(&v15, a3, a1);
    v7 = (unsigned int)v21;
    if ( (unsigned int)v21 >= HIDWORD(v21) )
    {
      v13 = v6;
      sub_16CD150(&v20, v22, 0, 8);
      v7 = (unsigned int)v21;
      v6 = v13;
    }
    *(_QWORD *)&v20[8 * v7] = v6;
    v8 = (unsigned int)v21;
    v9 = v21 + 1;
    LODWORD(v21) = v21 + 1;
  }
  while ( v16 );
  v10 = v20;
  v5 = v9;
  if ( !v9 )
  {
    v5 = 0;
    goto LABEL_7;
  }
  if ( *(_BYTE *)(*(_QWORD *)&v20[8 * v9 - 8] + 8LL) )
  {
LABEL_7:
    v11 = sub_1644EA0(v14, v10, v5, 0);
    goto LABEL_8;
  }
  LODWORD(v21) = v8;
  v11 = sub_1644EA0(v14, v20, v8, 1);
LABEL_8:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  if ( v17 != (unsigned int *)v19 )
    _libc_free((unsigned __int64)v17);
  return v11;
}
