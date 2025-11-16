// Function: sub_117F800
// Address: 0x117f800
//
bool __fastcall sub_117F800(const void ***a1, __int64 a2)
{
  _BYTE *v2; // r12
  const void **v3; // r13
  unsigned int v4; // edx
  unsigned int v5; // r8d
  bool result; // al
  __int64 v7; // rdx
  _BYTE *v8; // rax
  bool v9; // [rsp+Fh] [rbp-41h]
  const void *v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]

  v2 = (_BYTE *)a2;
  if ( *(_BYTE *)a2 != 17 )
  {
    v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v7 > 1 )
      return 0;
    if ( *(_BYTE *)a2 > 0x15u )
      return 0;
    v8 = sub_AD7630(a2, 1, v7);
    v2 = v8;
    if ( !v8 || *v8 != 17 )
      return 0;
  }
  v3 = *a1;
  v4 = *((_DWORD *)v2 + 8);
  v5 = *((_DWORD *)*a1 + 2);
  if ( v4 == v5 )
  {
    if ( v4 <= 0x40 )
      return *((_QWORD *)v2 + 3) == (_QWORD)*v3;
    else
      return sub_C43C50((__int64)(v2 + 24), *a1);
  }
  else
  {
    if ( v4 <= v5 )
    {
      sub_C449B0((__int64)&v10, (const void **)v2 + 3, v5);
      if ( v11 <= 0x40 )
        return v10 == *v3;
      result = sub_C43C50((__int64)&v10, v3);
      goto LABEL_6;
    }
    sub_C449B0((__int64)&v10, *a1, v4);
    if ( *((_DWORD *)v2 + 8) <= 0x40u )
      result = *((_QWORD *)v2 + 3) == (_QWORD)v10;
    else
      result = sub_C43C50((__int64)(v2 + 24), &v10);
    if ( v11 > 0x40 )
    {
LABEL_6:
      if ( v10 )
      {
        v9 = result;
        j_j___libc_free_0_0(v10);
        return v9;
      }
    }
  }
  return result;
}
