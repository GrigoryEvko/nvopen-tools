// Function: sub_C3F220
// Address: 0xc3f220
//
__int64 __fastcall sub_C3F220(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  _DWORD *v4; // rax
  _DWORD *v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // r14
  _QWORD *i; // r12
  unsigned int v11; // [rsp+0h] [rbp-C0h]
  __int64 v13; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-A8h]
  __int64 v15; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-98h]
  __int64 v17[4]; // [rsp+30h] [rbp-90h] BYREF
  _DWORD *v18; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+58h] [rbp-68h]
  _DWORD *v20; // [rsp+70h] [rbp-50h] BYREF
  _QWORD *v21; // [rsp+78h] [rbp-48h]

  sub_C3E660((__int64)&v20, (__int64)a1);
  v4 = sub_C33340();
  v5 = v4;
  if ( v4 == dword_3F65580 )
  {
    sub_C3C640(v17, (__int64)v4, &v20);
    if ( (unsigned int)v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    sub_C3E660((__int64)&v15, a3);
    sub_C3C640(&v20, (__int64)dword_3F65580, &v15);
    sub_C3E660((__int64)&v13, a2);
    sub_C3C640(&v18, (__int64)dword_3F65580, &v13);
  }
  else
  {
    sub_C3B160((__int64)v17, dword_3F65580, (__int64 *)&v20);
    if ( (unsigned int)v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    sub_C3E660((__int64)&v15, a3);
    sub_C3B160((__int64)&v20, dword_3F65580, &v15);
    sub_C3E660((__int64)&v13, a2);
    sub_C3B160((__int64)&v18, dword_3F65580, &v13);
  }
  if ( (_DWORD *)v17[0] == v5 )
    v11 = sub_C3F220(v17, &v18, &v20, a4);
  else
    v11 = sub_C3B3E0((__int64)v17, (__int64)&v18, (__int64)&v20, a4);
  if ( v18 == v5 )
    sub_969EE0((__int64)&v18);
  else
    sub_C338F0((__int64)&v18);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v20 == v5 )
    sub_969EE0((__int64)&v20);
  else
    sub_C338F0((__int64)&v20);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( (_DWORD *)v17[0] == v5 )
    sub_C3E660((__int64)&v18, (__int64)v17);
  else
    sub_C3A850((__int64)&v18, v17);
  sub_C3C640(&v20, (__int64)&unk_3F655A0, &v18);
  v6 = (_QWORD *)a1[1];
  if ( v6 )
  {
    v7 = &v6[3 * *(v6 - 1)];
    if ( v6 != v7 )
    {
      do
      {
        while ( 1 )
        {
          v7 -= 3;
          if ( v5 == (_DWORD *)*v7 )
            break;
          sub_C338F0((__int64)v7);
          if ( (_QWORD *)a1[1] == v7 )
            goto LABEL_24;
        }
        sub_969EE0((__int64)v7);
      }
      while ( (_QWORD *)a1[1] != v7 );
    }
LABEL_24:
    j_j_j___libc_free_0_0(v7 - 1);
  }
  sub_C3C840(a1, &v20);
  if ( v21 )
  {
    for ( i = &v21[3 * *(v21 - 1)]; v21 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v5 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v21 == i )
          goto LABEL_31;
      }
    }
LABEL_31:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( (_DWORD *)v17[0] == v5 )
    sub_969EE0((__int64)v17);
  else
    sub_C338F0((__int64)v17);
  return v11;
}
