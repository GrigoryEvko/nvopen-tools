// Function: sub_CA3300
// Address: 0xca3300
//
__int64 __fastcall sub_CA3300(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int v7; // r12d
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 *v11; // [rsp+0h] [rbp-80h] BYREF
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  char v13; // [rsp+20h] [rbp-60h]
  const char *v14[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v15; // [rsp+50h] [rbp-30h]

  v2 = *(unsigned __int8 **)a2;
  v15 = 261;
  v14[0] = (const char *)v2;
  v14[1] = *(const char **)(a2 + 8);
  if ( (unsigned __int8)sub_C81DB0(v14, 0) )
  {
    sub_2241E40(v14, 0, v3, v4, v5);
    return 0;
  }
  else
  {
    (*(void (__fastcall **)(__int64 **, __int64))(*(_QWORD *)a1 + 80LL))(&v11, a1);
    if ( (v13 & 1) != 0 )
    {
      return (unsigned int)v11;
    }
    else
    {
      v14[0] = (const char *)&v11;
      v15 = 260;
      v7 = 0;
      sub_C846B0((__int64)v14, (unsigned __int8 **)a2);
      sub_2241E40(v14, a2, v8, v9, v10);
      if ( (v13 & 1) == 0 && v11 != &v12 )
        j_j___libc_free_0(v11, v12 + 1);
    }
    return v7;
  }
}
