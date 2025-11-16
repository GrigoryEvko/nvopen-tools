// Function: sub_3368DE0
// Address: 0x3368de0
//
__int64 __fastcall sub_3368DE0(__int64 a1, unsigned int a2, __int64 a3)
{
  _DWORD *v4; // r12
  _DWORD *v5; // rax
  _DWORD *v6; // rbx
  __int64 v7; // r12
  _QWORD *i; // rbx
  unsigned __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-58h]
  _DWORD *v12; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+18h] [rbp-48h]

  v11 = 32;
  v10 = a2;
  v4 = sub_C33310();
  v5 = sub_C33340();
  v6 = v5;
  if ( v4 == v5 )
    sub_C3C640(&v12, (__int64)v5, &v10);
  else
    sub_C3B160((__int64)&v12, v4, (__int64 *)&v10);
  v7 = sub_33FE6E0(a1, &v12, a3, 12, 0, 0);
  if ( v12 == v6 )
  {
    if ( v13 )
    {
      for ( i = &v13[3 * *(v13 - 1)]; v13 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v12);
  }
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return v7;
}
