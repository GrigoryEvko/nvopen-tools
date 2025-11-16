// Function: sub_25533A0
// Address: 0x25533a0
//
const void *__fastcall sub_25533A0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  unsigned __int64 v4; // r14
  bool v5; // r13
  __int64 **v6; // r13
  unsigned __int64 v7; // rax
  const void *v8; // rax
  const void *v10; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-68h]
  const void *v12; // [rsp+20h] [rbp-60h] BYREF
  __int64 v13; // [rsp+28h] [rbp-58h]
  const void *v14; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+38h] [rbp-48h]
  const void *v16; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+48h] [rbp-38h]

  (*(void (__fastcall **)(const void **, _QWORD *, __int64, __int64))(*a1 + 112LL))(&v14, a1, a2, a3);
  v11 = v15;
  if ( v15 > 0x40 )
    sub_C43780((__int64)&v10, &v14);
  else
    v10 = v14;
  sub_C46A40((__int64)&v10, 1);
  v3 = v11;
  v4 = (unsigned __int64)v10;
  v11 = 0;
  LODWORD(v13) = v3;
  v12 = v10;
  if ( v17 <= 0x40 )
    v5 = v16 == v10;
  else
    v5 = sub_C43C50((__int64)&v16, &v12);
  if ( v3 > 0x40 )
  {
    if ( v4 )
    {
      j_j___libc_free_0_0(v4);
      if ( v11 > 0x40 )
      {
        if ( v10 )
          j_j___libc_free_0_0((unsigned __int64)v10);
      }
    }
  }
  if ( v5 )
  {
    v6 = *(__int64 ***)(sub_250D070(a1 + 9) + 8);
    v7 = sub_ACCFD0(*v6, (__int64)&v14);
    v8 = (const void *)sub_250C3F0(v7, (__int64)v6);
    LOBYTE(v13) = 1;
    v12 = v8;
  }
  else if ( sub_AAF7D0((__int64)&v14) )
  {
    LOBYTE(v13) = 0;
  }
  else
  {
    v12 = 0;
    LOBYTE(v13) = 1;
  }
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0((unsigned __int64)v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0((unsigned __int64)v14);
  return v12;
}
