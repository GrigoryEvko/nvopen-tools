// Function: sub_16A6CD0
// Address: 0x16a6cd0
//
unsigned __int64 __fastcall sub_16A6CD0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v3; // r12d
  unsigned __int64 v4; // r10
  __int64 *v5; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v7; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v8[3]; // [rsp+18h] [rbp-18h] BYREF

  v2 = *(_DWORD *)(a1 + 8);
  v7 = a2;
  if ( v2 <= 0x40 )
    return *(_QWORD *)a1 % a2;
  v3 = v2 - sub_16A57B0(a1);
  v4 = ((unsigned __int64)v3 + 63) >> 6;
  if ( !v4 )
    return v4;
  if ( v7 == 1 )
    return 0;
  v5 = *(__int64 **)a1;
  if ( v3 > 0x40 )
    goto LABEL_7;
  result = *v5;
  if ( v7 <= *v5 )
  {
    if ( v7 != *v5 )
    {
LABEL_7:
      if ( v4 == 1 )
        return *v5 % v7;
      sub_16A6110(v5, v4, (__int64 *)&v7, 1u, 0, (__int64)v8);
      return v8[0];
    }
    return 0;
  }
  return result;
}
