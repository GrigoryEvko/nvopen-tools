// Function: sub_2C2C640
// Address: 0x2c2c640
//
bool __fastcall sub_2C2C640(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r13
  unsigned int v4; // edx
  unsigned int v5; // r8d
  bool v6; // al
  __int64 v7; // rdx
  _BYTE *v8; // rax
  bool v9; // [rsp+Fh] [rbp-41h]
  const void *v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]

  if ( sub_2BF04A0(a2) )
    return 0;
  v3 = *(_BYTE **)(a2 + 40);
  if ( !v3 )
    return 0;
  if ( *v3 != 17 )
  {
    v7 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v3 + 1) + 8LL) - 17;
    if ( (unsigned int)v7 > 1 )
      return 0;
    if ( *v3 > 0x15u )
      return 0;
    v8 = sub_AD7630(*(_QWORD *)(a2 + 40), 0, v7);
    v3 = v8;
    if ( !v8 || *v8 != 17 )
      return 0;
  }
  v4 = *((_DWORD *)v3 + 8);
  v5 = *(_DWORD *)(a1 + 8);
  if ( v4 == v5 )
  {
    if ( v4 <= 0x40 )
      return *((_QWORD *)v3 + 3) == *(_QWORD *)a1;
    else
      return sub_C43C50((__int64)(v3 + 24), (const void **)a1);
  }
  else
  {
    if ( v4 <= v5 )
    {
      sub_C449B0((__int64)&v10, (const void **)v3 + 3, v5);
      if ( v11 <= 0x40 )
        v6 = v10 == *(const void **)a1;
      else
        v6 = sub_C43C50((__int64)&v10, (const void **)a1);
    }
    else
    {
      sub_C449B0((__int64)&v10, (const void **)a1, v4);
      if ( *((_DWORD *)v3 + 8) <= 0x40u )
        v6 = *((_QWORD *)v3 + 3) == (_QWORD)v10;
      else
        v6 = sub_C43C50((__int64)(v3 + 24), &v10);
    }
    v9 = v6;
    sub_969240((__int64 *)&v10);
    return v9;
  }
}
