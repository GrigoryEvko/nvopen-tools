// Function: sub_C4B670
// Address: 0xc4b670
//
__int64 __fastcall sub_C4B670(unsigned int a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 *v3; // r15
  unsigned int v4; // r12d
  __int64 result; // rax
  unsigned int v6; // eax
  unsigned int v7; // [rsp-6Ch] [rbp-6Ch]
  __int64 *v8; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v9; // [rsp-60h] [rbp-60h]
  __int64 v10; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v11; // [rsp-50h] [rbp-50h]
  __int64 *v12; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v13; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return 0;
  v2 = *(_DWORD *)(a2 + 8);
  v9 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780((__int64)&v8, (const void **)a2);
    if ( a1 <= v2 )
    {
      v6 = v9;
      goto LABEL_19;
    }
  }
  else
  {
    v8 = *(__int64 **)a2;
    if ( a1 <= v2 )
    {
      v11 = v2;
      goto LABEL_5;
    }
  }
  sub_C449B0((__int64)&v12, (const void **)a2, a1);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  v8 = v12;
  v6 = v13;
  v9 = v13;
LABEL_19:
  v11 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690((__int64)&v10, a1, 0);
    goto LABEL_6;
  }
LABEL_5:
  v10 = a1;
LABEL_6:
  sub_C4B490((__int64)&v12, (__int64)&v8, (__int64)&v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  v3 = v12;
  v4 = v13;
  v8 = v12;
  v9 = v13;
  if ( v11 > 0x40 && v10 )
  {
    j_j___libc_free_0_0(v10);
    v4 = v9;
    v3 = v8;
  }
  if ( v4 > 0x40 )
  {
    if ( v4 - (unsigned int)sub_C444A0((__int64)&v8) <= 0x40 )
    {
      result = *v3;
      if ( *v3 > (unsigned __int64)a1 )
      {
        j_j___libc_free_0_0(v3);
        return a1;
      }
    }
    else
    {
      result = a1;
    }
    if ( v3 )
    {
      v7 = result;
      j_j___libc_free_0_0(v3);
      return v7;
    }
  }
  else
  {
    result = (unsigned int)v3;
    if ( (unsigned __int64)v3 > a1 )
      return a1;
  }
  return result;
}
