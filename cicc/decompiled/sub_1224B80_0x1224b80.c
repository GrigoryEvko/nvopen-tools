// Function: sub_1224B80
// Address: 0x1224b80
//
__int64 __fastcall sub_1224B80(__int64 **a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  unsigned int v5; // r12d
  void *v6; // r15
  void **i; // rbx
  _DWORD *v10; // [rsp+10h] [rbp-110h]
  __int64 v11[4]; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v12; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v13; // [rsp+58h] [rbp-C8h]
  __int64 v14; // [rsp+68h] [rbp-B8h]
  _QWORD *v15; // [rsp+70h] [rbp-B0h]
  __int64 v16; // [rsp+78h] [rbp-A8h]
  _QWORD v17[2]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD *v18; // [rsp+90h] [rbp-90h]
  __int64 v19; // [rsp+98h] [rbp-88h]
  _QWORD v20[2]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v21; // [rsp+B0h] [rbp-70h]
  unsigned int v22; // [rsp+B8h] [rbp-68h]
  char v23; // [rsp+BCh] [rbp-64h]
  void *v24; // [rsp+C0h] [rbp-60h] BYREF
  void **v25; // [rsp+C8h] [rbp-58h]
  __int64 v26; // [rsp+E0h] [rbp-40h]
  char v27; // [rsp+E8h] [rbp-38h]

  *a3 = 0;
  v15 = v17;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  v18 = v20;
  v19 = 0;
  LOBYTE(v20[0]) = 0;
  v22 = 1;
  v21 = 0;
  v23 = 0;
  v10 = sub_C33320();
  sub_C3B1B0((__int64)v11, 0.0);
  sub_C407B0(&v24, v11, v10);
  sub_C338F0((__int64)v11);
  v26 = 0;
  v27 = 0;
  v5 = sub_1221570(a1, (__int64)&v12, (__int64)a4, a2);
  if ( !(_BYTE)v5 )
    v5 = sub_121E800(a1, a2, &v12, a3, a4, (int)a3);
  if ( v26 )
    j_j___libc_free_0_0(v26);
  v6 = sub_C33340();
  if ( v24 == v6 )
  {
    if ( v25 )
    {
      for ( i = &v25[3 * (_QWORD)*(v25 - 1)]; v25 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v6 == *i )
            break;
          sub_C338F0((__int64)i);
          if ( v25 == i )
            goto LABEL_17;
        }
      }
LABEL_17:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v24);
  }
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v18 != v20 )
    j_j___libc_free_0(v18, v20[0] + 1LL);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  return v5;
}
