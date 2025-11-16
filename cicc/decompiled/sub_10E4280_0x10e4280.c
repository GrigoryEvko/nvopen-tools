// Function: sub_10E4280
// Address: 0x10e4280
//
bool __fastcall sub_10E4280(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v4; // rax
  _BYTE *v5; // r12
  __int64 v6; // r13
  unsigned int v7; // edx
  unsigned int v8; // r8d
  __int64 v9; // rdx
  _BYTE *v10; // rax
  bool v11; // [rsp-49h] [rbp-49h]
  const void *v12; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v13; // [rsp-40h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
  **(_QWORD **)a1 = v4;
  v5 = *(_BYTE **)(a2 - 32);
  if ( !v5 )
    BUG();
  if ( *v5 != 17 )
  {
    v9 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v5 + 1) + 8LL) - 17;
    if ( (unsigned int)v9 > 1 )
      return 0;
    if ( *v5 > 0x15u )
      return 0;
    v10 = sub_AD7630(*(_QWORD *)(a2 - 32), 0, v9);
    v5 = v10;
    if ( !v10 || *v10 != 17 )
      return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *((_DWORD *)v5 + 8);
  v8 = *(_DWORD *)(v6 + 8);
  if ( v7 == v8 )
  {
    if ( v7 <= 0x40 )
      return *((_QWORD *)v5 + 3) == *(_QWORD *)v6;
    else
      return sub_C43C50((__int64)(v5 + 24), *(const void ***)(a1 + 8));
  }
  else
  {
    if ( v7 <= v8 )
    {
      sub_C449B0((__int64)&v12, (const void **)v5 + 3, v8);
      if ( v13 <= 0x40 )
        return v12 == *(const void **)v6;
      result = sub_C43C50((__int64)&v12, (const void **)v6);
      goto LABEL_13;
    }
    sub_C449B0((__int64)&v12, *(const void ***)(a1 + 8), v7);
    if ( *((_DWORD *)v5 + 8) <= 0x40u )
      result = *((_QWORD *)v5 + 3) == (_QWORD)v12;
    else
      result = sub_C43C50((__int64)(v5 + 24), &v12);
    if ( v13 > 0x40 )
    {
LABEL_13:
      if ( v12 )
      {
        v11 = result;
        j_j___libc_free_0_0(v12);
        return v11;
      }
    }
  }
  return result;
}
