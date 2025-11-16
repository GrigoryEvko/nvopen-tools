// Function: sub_1160300
// Address: 0x1160300
//
bool __fastcall sub_1160300(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edx
  __int64 v5; // rax
  _BYTE *v6; // r12
  __int64 v7; // r13
  unsigned int v8; // edx
  unsigned int v9; // r8d
  bool result; // al
  __int64 v11; // rdx
  _BYTE *v12; // rax
  bool v13; // [rsp+Fh] [rbp-41h]
  const void *v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]

  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x1Cu )
  {
    if ( (_BYTE)v2 == 5 )
    {
      v4 = *((unsigned __int16 *)a2 + 1);
      if ( (*((_WORD *)a2 + 1) & 0xFFFD) == 0xD || (v4 & 0xFFF7) == 0x11 )
        goto LABEL_4;
    }
    return 0;
  }
  if ( (unsigned __int8)v2 > 0x36u )
    return 0;
  v3 = 0x40540000000000LL;
  v4 = (unsigned __int8)v2 - 29;
  if ( !_bittest64(&v3, v2) )
    return 0;
LABEL_4:
  if ( v4 != 17 )
    return 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v5 = *((_QWORD *)a2 - 8);
  if ( !v5 )
    return 0;
  **(_QWORD **)a1 = v5;
  v6 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( !v6 )
    BUG();
  if ( *v6 != 17 )
  {
    v11 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v6 + 1) + 8LL) - 17;
    if ( (unsigned int)v11 > 1 )
      return 0;
    if ( *v6 > 0x15u )
      return 0;
    v12 = sub_AD7630(*((_QWORD *)a2 - 4), 0, v11);
    v6 = v12;
    if ( !v12 || *v12 != 17 )
      return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *((_DWORD *)v6 + 8);
  v9 = *(_DWORD *)(v7 + 8);
  if ( v8 == v9 )
  {
    if ( v8 <= 0x40 )
      return *((_QWORD *)v6 + 3) == *(_QWORD *)v7;
    else
      return sub_C43C50((__int64)(v6 + 24), *(const void ***)(a1 + 8));
  }
  else
  {
    if ( v8 <= v9 )
    {
      sub_C449B0((__int64)&v14, (const void **)v6 + 3, v9);
      if ( v15 <= 0x40 )
        return v14 == *(const void **)v7;
      result = sub_C43C50((__int64)&v14, (const void **)v7);
      goto LABEL_13;
    }
    sub_C449B0((__int64)&v14, *(const void ***)(a1 + 8), v8);
    if ( *((_DWORD *)v6 + 8) <= 0x40u )
      result = *((_QWORD *)v6 + 3) == (_QWORD)v14;
    else
      result = sub_C43C50((__int64)(v6 + 24), &v14);
    if ( v15 > 0x40 )
    {
LABEL_13:
      if ( v14 )
      {
        v13 = result;
        j_j___libc_free_0_0(v14);
        return v13;
      }
    }
  }
  return result;
}
