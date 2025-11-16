// Function: sub_C3EF50
// Address: 0xc3ef50
//
__int64 __fastcall sub_C3EF50(_QWORD *a1, __int64 a2, unsigned __int8 a3)
{
  _DWORD *v3; // rax
  _DWORD *v4; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // r14
  _QWORD *i; // r12
  unsigned int v9; // [rsp+0h] [rbp-90h]
  __int64 v11; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-78h]
  __int64 v13[4]; // [rsp+20h] [rbp-70h] BYREF
  _DWORD *v14; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+48h] [rbp-48h]

  sub_C3E660((__int64)&v14, (__int64)a1);
  v3 = sub_C33340();
  v4 = v3;
  if ( v3 == dword_3F65580 )
  {
    sub_C3C640(v13, (__int64)v3, &v14);
    if ( (unsigned int)v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    sub_C3E660((__int64)&v11, a2);
    sub_C3C640(&v14, (__int64)dword_3F65580, &v11);
  }
  else
  {
    sub_C3B160((__int64)v13, dword_3F65580, (__int64 *)&v14);
    if ( (unsigned int)v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    sub_C3E660((__int64)&v11, a2);
    sub_C3B160((__int64)&v14, dword_3F65580, &v11);
  }
  if ( (_DWORD *)v13[0] == v4 )
    v9 = sub_C3EF50(v13, &v14, a3);
  else
    v9 = sub_C3B6C0((__int64)v13, (__int64)&v14, a3);
  if ( v14 == v4 )
    sub_969EE0((__int64)&v14);
  else
    sub_C338F0((__int64)&v14);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( (_DWORD *)v13[0] == v4 )
    sub_C3E660((__int64)&v11, (__int64)v13);
  else
    sub_C3A850((__int64)&v11, v13);
  sub_C3C640(&v14, (__int64)&unk_3F655A0, &v11);
  v5 = (_QWORD *)a1[1];
  if ( v5 )
  {
    v6 = &v5[3 * *(v5 - 1)];
    if ( v5 != v6 )
    {
      do
      {
        while ( 1 )
        {
          v6 -= 3;
          if ( v4 == (_DWORD *)*v6 )
            break;
          sub_C338F0((__int64)v6);
          if ( (_QWORD *)a1[1] == v6 )
            goto LABEL_19;
        }
        sub_969EE0((__int64)v6);
      }
      while ( (_QWORD *)a1[1] != v6 );
    }
LABEL_19:
    j_j_j___libc_free_0_0(v6 - 1);
  }
  sub_C3C840(a1, &v14);
  if ( v15 )
  {
    for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v4 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v15 == i )
          goto LABEL_26;
      }
    }
LABEL_26:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( (_DWORD *)v13[0] == v4 )
    sub_969EE0((__int64)v13);
  else
    sub_C338F0((__int64)v13);
  return v9;
}
