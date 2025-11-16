// Function: sub_31D9E60
// Address: 0x31d9e60
//
__int64 __fastcall sub_31D9E60(unsigned __int8 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r14
  char v4; // bl
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v10; // rbx
  unsigned int v11; // ecx
  unsigned int v12; // edx
  unsigned __int8 *v13; // rax
  int v14; // edx
  int v15; // esi
  unsigned int v16; // edx
  unsigned __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  unsigned int *v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h]

  v2 = *a1;
  if ( (_BYTE)v2 != 17 )
  {
    if ( (_BYTE)v2 == 9 )
    {
      v10 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v5 = sub_31D9E60(v10);
      if ( v5 != -1 )
      {
        v11 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
        if ( v11 == 1 )
          return v5;
        v12 = 1;
        while ( v10 == *(_QWORD *)&a1[32 * (v12 - (unsigned __int64)v11)] )
        {
          if ( ++v12 == v11 )
            return v5;
        }
      }
    }
    else if ( (unsigned int)(v2 - 15) <= 1 )
    {
      v13 = (unsigned __int8 *)sub_AC52D0((__int64)a1);
      v5 = *v13;
      v15 = v14;
      if ( v14 == 1 )
        return v5;
      v16 = 1;
      while ( (_BYTE)v5 == v13[v16] )
      {
        if ( v15 == ++v16 )
          return v5;
      }
    }
    return (unsigned int)-1;
  }
  v3 = *((_QWORD *)a1 + 1);
  v4 = sub_AE5020(a2, v3);
  v5 = -1;
  v6 = sub_9208B0(a2, v3);
  v20 = v7;
  v19 = (unsigned int *)(8 * (((1LL << v4) + ((unsigned __int64)(v6 + 7) >> 3) - 1) >> v4 << v4));
  v8 = sub_CA1930(&v19);
  sub_C449B0((__int64)&v17, (const void **)a1 + 3, v8);
  if ( sub_C489C0((__int64)&v17, 8u)
    && (sub_C44AB0((__int64)&v19, (__int64)&v17, 8u), v5 = (unsigned int)v19, (unsigned int)v20 > 0x40) )
  {
    v5 = *v19;
    j_j___libc_free_0_0((unsigned __int64)v19);
    if ( v18 <= 0x40 )
      return v5;
  }
  else if ( v18 <= 0x40 )
  {
    return v5;
  }
  if ( v17 )
    j_j___libc_free_0_0(v17);
  return v5;
}
