// Function: sub_1224770
// Address: 0x1224770
//
__int64 __fastcall sub_1224770(__int64 **a1, __int64 a2, _QWORD *a3)
{
  _DWORD *v3; // r12
  unsigned int v4; // eax
  _QWORD *v5; // r9
  unsigned int v6; // r12d
  void *v7; // r15
  unsigned int v9; // eax
  void **i; // rbx
  _BYTE *v12; // [rsp+28h] [rbp-108h] BYREF
  __int64 v13[4]; // [rsp+30h] [rbp-100h] BYREF
  char v14; // [rsp+50h] [rbp-E0h]
  char v15; // [rsp+51h] [rbp-DFh]
  unsigned int v16; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int64 v17; // [rsp+68h] [rbp-C8h]
  __int64 v18; // [rsp+78h] [rbp-B8h]
  _QWORD *v19; // [rsp+80h] [rbp-B0h]
  __int64 v20; // [rsp+88h] [rbp-A8h]
  _QWORD v21[2]; // [rsp+90h] [rbp-A0h] BYREF
  _QWORD *v22; // [rsp+A0h] [rbp-90h]
  __int64 v23; // [rsp+A8h] [rbp-88h]
  _QWORD v24[2]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v25; // [rsp+C0h] [rbp-70h]
  unsigned int v26; // [rsp+C8h] [rbp-68h]
  char v27; // [rsp+CCh] [rbp-64h]
  void *v28; // [rsp+D0h] [rbp-60h] BYREF
  void **v29; // [rsp+D8h] [rbp-58h]
  __int64 v30; // [rsp+F0h] [rbp-40h]
  char v31; // [rsp+F8h] [rbp-38h]

  *a3 = 0;
  v19 = v21;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v20 = 0;
  LOBYTE(v21[0]) = 0;
  v22 = v24;
  v23 = 0;
  LOBYTE(v24[0]) = 0;
  v26 = 1;
  v25 = 0;
  v27 = 0;
  v3 = sub_C33320();
  sub_C3B1B0((__int64)v13, 0.0);
  sub_C407B0(&v28, v13, v3);
  sub_C338F0((__int64)v13);
  v31 = 0;
  v30 = 0;
  v12 = 0;
  v4 = sub_1221570(a1, (__int64)&v16, 0, a2);
  v5 = a3;
  v6 = v4;
  if ( !(_BYTE)v4 )
  {
    v9 = sub_121E800(a1, a2, &v16, (__int64 *)&v12, 0, (int)a3);
    v5 = a3;
    v6 = v9;
  }
  if ( v12 )
  {
    if ( *v12 > 0x15u )
    {
      *v5 = 0;
      v15 = 1;
      v6 = 1;
      v13[0] = (__int64)"global values must be constants";
      v14 = 3;
      sub_11FD800((__int64)(a1 + 22), v17, (__int64)v13, 1);
    }
    else
    {
      *v5 = v12;
    }
  }
  if ( v30 )
    j_j___libc_free_0_0(v30);
  v7 = sub_C33340();
  if ( v28 == v7 )
  {
    if ( v29 )
    {
      for ( i = &v29[3 * (_QWORD)*(v29 - 1)]; v29 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v7 == *i )
            break;
          sub_C338F0((__int64)i);
          if ( v29 == i )
            goto LABEL_21;
        }
      }
LABEL_21:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v28);
  }
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v22 != v24 )
    j_j___libc_free_0(v22, v24[0] + 1LL);
  if ( v19 != v21 )
    j_j___libc_free_0(v19, v21[0] + 1LL);
  return v6;
}
