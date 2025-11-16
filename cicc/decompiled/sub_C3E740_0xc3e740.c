// Function: sub_C3E740
// Address: 0xc3e740
//
__int64 __fastcall sub_C3E740(_QWORD *a1, unsigned int a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  _QWORD *i; // r12
  _QWORD *j; // r12
  unsigned int v9; // [rsp+4h] [rbp-7Ch]
  __int64 v10; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-68h]
  __int64 v12; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+28h] [rbp-58h]
  _DWORD *v14; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+38h] [rbp-48h]

  sub_C3E660((__int64)&v12, (__int64)a1);
  v2 = sub_C33340();
  v3 = v2;
  if ( v2 == dword_3F65580 )
    sub_C3C640(&v14, (__int64)v2, &v12);
  else
    sub_C3B160((__int64)&v14, dword_3F65580, &v12);
  if ( (unsigned int)v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v14 == v3 )
    v9 = sub_C3E740(&v14, a2);
  else
    v9 = sub_C3BAB0((__int64)&v14, a2);
  if ( v14 == v3 )
    sub_C3E660((__int64)&v10, (__int64)&v14);
  else
    sub_C3A850((__int64)&v10, (__int64 *)&v14);
  sub_C3C640(&v12, (__int64)&unk_3F655A0, &v10);
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
            goto LABEL_16;
        }
        sub_969EE0((__int64)v5);
      }
      while ( (_QWORD *)a1[1] != v5 );
    }
LABEL_16:
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
          goto LABEL_23;
      }
    }
LABEL_23:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v14 == v3 )
  {
    if ( v15 )
    {
      for ( j = &v15[3 * *(v15 - 1)]; v15 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v3 == (_DWORD *)*j )
            break;
          sub_C338F0((__int64)j);
          if ( v15 == j )
            goto LABEL_35;
        }
      }
LABEL_35:
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v14);
  }
  return v9;
}
