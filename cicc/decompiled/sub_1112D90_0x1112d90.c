// Function: sub_1112D90
// Address: 0x1112d90
//
bool __fastcall sub_1112D90(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  unsigned int v3; // r8d
  bool result; // al
  bool v5; // [rsp+Fh] [rbp-31h]
  const void *v6; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-28h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  if ( v2 == v3 )
  {
    if ( v2 <= 0x40 )
      return *(_QWORD *)a1 == *(_QWORD *)a2;
    else
      return sub_C43C50(a1, (const void **)a2);
  }
  else
  {
    if ( v2 <= v3 )
    {
      sub_C449B0((__int64)&v6, (const void **)a1, v3);
      if ( v7 <= 0x40 )
        return v6 == *(const void **)a2;
      result = sub_C43C50((__int64)&v6, (const void **)a2);
      goto LABEL_5;
    }
    sub_C449B0((__int64)&v6, (const void **)a2, v2);
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
      result = *(_QWORD *)a1 == (_QWORD)v6;
    else
      result = sub_C43C50(a1, &v6);
    if ( v7 > 0x40 )
    {
LABEL_5:
      if ( v6 )
      {
        v5 = result;
        j_j___libc_free_0_0(v6);
        return v5;
      }
    }
  }
  return result;
}
