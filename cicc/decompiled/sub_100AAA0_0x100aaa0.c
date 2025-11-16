// Function: sub_100AAA0
// Address: 0x100aaa0
//
bool __fastcall sub_100AAA0(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // edx
  unsigned int v7; // r8d
  __int64 v8; // rdx
  _BYTE *v9; // rax
  bool v10; // [rsp-49h] [rbp-49h]
  const void *v11; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v12; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  if ( *(_QWORD *)a1 != *((_QWORD *)a3 - 8) )
    return 0;
  v4 = *((_QWORD *)a3 - 4);
  if ( !v4 )
    BUG();
  if ( *(_BYTE *)v4 != 17 )
  {
    v8 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
    if ( (unsigned int)v8 > 1 )
      return 0;
    if ( *(_BYTE *)v4 > 0x15u )
      return 0;
    v9 = sub_AD7630(v4, 0, v8);
    v4 = (__int64)v9;
    if ( !v9 || *v9 != 17 )
      return 0;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_DWORD *)(v4 + 32);
  v7 = *(_DWORD *)(v5 + 8);
  if ( v6 == v7 )
  {
    if ( v6 <= 0x40 )
      return *(_QWORD *)(v4 + 24) == *(_QWORD *)v5;
    else
      return sub_C43C50(v4 + 24, *(const void ***)(a1 + 8));
  }
  else
  {
    if ( v6 <= v7 )
    {
      sub_C449B0((__int64)&v11, (const void **)(v4 + 24), v7);
      if ( v12 <= 0x40 )
        return v11 == *(const void **)v5;
      result = sub_C43C50((__int64)&v11, (const void **)v5);
      goto LABEL_13;
    }
    sub_C449B0((__int64)&v11, *(const void ***)(a1 + 8), v6);
    if ( *(_DWORD *)(v4 + 32) <= 0x40u )
      result = *(_QWORD *)(v4 + 24) == (_QWORD)v11;
    else
      result = sub_C43C50(v4 + 24, &v11);
    if ( v12 > 0x40 )
    {
LABEL_13:
      if ( v11 )
      {
        v10 = result;
        j_j___libc_free_0_0(v11);
        return v10;
      }
    }
  }
  return result;
}
