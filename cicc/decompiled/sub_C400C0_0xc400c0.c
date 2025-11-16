// Function: sub_C400C0
// Address: 0xc400c0
//
__int64 __fastcall sub_C400C0(_QWORD *a1, __int64 a2, unsigned __int8 a3, unsigned int a4)
{
  _DWORD *v5; // rax
  _DWORD *v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r15
  _QWORD *i; // r12
  _QWORD *j; // r12
  unsigned int v13; // [rsp+Ch] [rbp-74h]
  __int64 v14; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v17; // [rsp+28h] [rbp-58h]
  _DWORD *v18; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v19; // [rsp+38h] [rbp-48h]

  v5 = sub_C33340();
  v6 = v5;
  if ( v5 == dword_3F65580 )
    sub_C3C460(&v18, (__int64)v5);
  else
    sub_C37380(&v18, (__int64)dword_3F65580);
  if ( v18 == v6 )
    v13 = sub_C400C0(&v18, a2, a3, a4);
  else
    v13 = sub_C36910((__int64)&v18, a2, a3, a4);
  if ( v18 == v6 )
    sub_C3E660((__int64)&v14, (__int64)&v18);
  else
    sub_C3A850((__int64)&v14, (__int64 *)&v18);
  sub_C3C640(&v16, (__int64)&unk_3F655A0, &v14);
  v7 = (_QWORD *)a1[1];
  if ( v7 )
  {
    v8 = &v7[3 * *(v7 - 1)];
    if ( v7 != v8 )
    {
      do
      {
        while ( 1 )
        {
          v8 -= 3;
          if ( v6 == (_DWORD *)*v8 )
            break;
          sub_C338F0((__int64)v8);
          if ( (_QWORD *)a1[1] == v8 )
            goto LABEL_13;
        }
        sub_969EE0((__int64)v8);
      }
      while ( (_QWORD *)a1[1] != v8 );
    }
LABEL_13:
    j_j_j___libc_free_0_0(v8 - 1);
  }
  sub_C3C840(a1, &v16);
  if ( v17 )
  {
    for ( i = &v17[3 * *(v17 - 1)]; v17 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v6 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v17 == i )
          goto LABEL_20;
      }
    }
LABEL_20:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v18 == v6 )
  {
    if ( v19 )
    {
      for ( j = &v19[3 * *(v19 - 1)]; v19 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v6 == (_DWORD *)*j )
            break;
          sub_C338F0((__int64)j);
          if ( v19 == j )
            goto LABEL_32;
        }
      }
LABEL_32:
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v18);
  }
  return v13;
}
