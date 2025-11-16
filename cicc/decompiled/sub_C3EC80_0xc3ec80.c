// Function: sub_C3EC80
// Address: 0xc3ec80
//
__int64 __fastcall sub_C3EC80(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rax
  _QWORD *v7; // r14
  _QWORD *i; // r12
  unsigned int v10; // [rsp+4h] [rbp-8Ch]
  __int64 v11; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-78h]
  __int64 v13[4]; // [rsp+20h] [rbp-70h] BYREF
  _DWORD *v14; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+48h] [rbp-48h]

  sub_C3E660((__int64)&v14, (__int64)a1);
  v2 = sub_C33340();
  v3 = v2;
  if ( v2 != dword_3F65580 )
  {
    sub_C3B160((__int64)v13, dword_3F65580, (__int64 *)&v14);
    if ( (unsigned int)v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    sub_C3E660((__int64)&v11, a2);
    sub_C3B160((__int64)&v14, dword_3F65580, &v11);
    if ( (_DWORD *)v13[0] != v3 )
      goto LABEL_4;
LABEL_38:
    v10 = sub_C3EC80(v13, &v14, v4, v5);
    if ( v14 != v3 )
      goto LABEL_5;
LABEL_39:
    sub_969EE0((__int64)&v14);
    goto LABEL_6;
  }
  sub_C3C640(v13, (__int64)v2, &v14);
  if ( (unsigned int)v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  sub_C3E660((__int64)&v11, a2);
  sub_C3C640(&v14, (__int64)dword_3F65580, &v11);
  if ( (_DWORD *)v13[0] == v3 )
    goto LABEL_38;
LABEL_4:
  v10 = sub_C3BE30(v13, (__int64 *)&v14);
  if ( v14 == v3 )
    goto LABEL_39;
LABEL_5:
  sub_C338F0((__int64)&v14);
LABEL_6:
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( (_DWORD *)v13[0] == v3 )
    sub_C3E660((__int64)&v11, (__int64)v13);
  else
    sub_C3A850((__int64)&v11, v13);
  sub_C3C640(&v14, (__int64)&unk_3F655A0, &v11);
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
          if ( v3 == (_DWORD *)*v7 )
            break;
          sub_C338F0((__int64)v7);
          if ( (_QWORD *)a1[1] == v7 )
            goto LABEL_17;
        }
        sub_969EE0((__int64)v7);
      }
      while ( (_QWORD *)a1[1] != v7 );
    }
LABEL_17:
    j_j_j___libc_free_0_0(v7 - 1);
  }
  sub_C3C840(a1, &v14);
  if ( v15 )
  {
    for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v3 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v15 == i )
          goto LABEL_24;
      }
    }
LABEL_24:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( (_DWORD *)v13[0] == v3 )
    sub_969EE0((__int64)v13);
  else
    sub_C338F0((__int64)v13);
  return v10;
}
