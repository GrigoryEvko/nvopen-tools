// Function: sub_C3E9B0
// Address: 0xc3e9b0
//
__int64 __fastcall sub_C3E9B0(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // r14
  _QWORD *i; // r12
  unsigned int v8; // [rsp+4h] [rbp-8Ch]
  __int64 v9; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-78h]
  __int64 v11[4]; // [rsp+20h] [rbp-70h] BYREF
  _DWORD *v12; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+48h] [rbp-48h]

  sub_C3E660((__int64)&v12, (__int64)a1);
  v2 = sub_C33340();
  v3 = v2;
  if ( v2 != dword_3F65580 )
  {
    sub_C3B160((__int64)v11, dword_3F65580, (__int64 *)&v12);
    if ( (unsigned int)v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    sub_C3E660((__int64)&v9, a2);
    sub_C3B160((__int64)&v12, dword_3F65580, &v9);
    if ( (_DWORD *)v11[0] != v3 )
      goto LABEL_4;
LABEL_38:
    v8 = sub_C3E9B0(v11, &v12);
    if ( v12 != v3 )
      goto LABEL_5;
LABEL_39:
    sub_969EE0((__int64)&v12);
    goto LABEL_6;
  }
  sub_C3C640(v11, (__int64)v2, &v12);
  if ( (unsigned int)v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  sub_C3E660((__int64)&v9, a2);
  sub_C3C640(&v12, (__int64)dword_3F65580, &v9);
  if ( (_DWORD *)v11[0] == v3 )
    goto LABEL_38;
LABEL_4:
  v8 = sub_C3C0A0(v11, (__int64 *)&v12);
  if ( v12 == v3 )
    goto LABEL_39;
LABEL_5:
  sub_C338F0((__int64)&v12);
LABEL_6:
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( (_DWORD *)v11[0] == v3 )
    sub_C3E660((__int64)&v9, (__int64)v11);
  else
    sub_C3A850((__int64)&v9, v11);
  sub_C3C640(&v12, (__int64)&unk_3F655A0, &v9);
  v4 = (_QWORD *)a1[1];
  if ( v4 )
  {
    v5 = &v4[3 * *(v4 - 1)];
    if ( v4 != v5 )
    {
      do
      {
        while ( 1 )
        {
          v5 -= 3;
          if ( v3 == (_DWORD *)*v5 )
            break;
          sub_C338F0((__int64)v5);
          if ( (_QWORD *)a1[1] == v5 )
            goto LABEL_17;
        }
        sub_969EE0((__int64)v5);
      }
      while ( (_QWORD *)a1[1] != v5 );
    }
LABEL_17:
    j_j_j___libc_free_0_0(v5 - 1);
  }
  sub_C3C840(a1, &v12);
  if ( v13 )
  {
    for ( i = &v13[3 * *(v13 - 1)]; v13 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v3 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v13 == i )
          goto LABEL_24;
      }
    }
LABEL_24:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( (_DWORD *)v11[0] == v3 )
    sub_969EE0((__int64)v11);
  else
    sub_C338F0((__int64)v11);
  return v8;
}
