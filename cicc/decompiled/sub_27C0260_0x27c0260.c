// Function: sub_27C0260
// Address: 0x27c0260
//
__int64 __fastcall sub_27C0260(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rcx

  v2 = *a1;
  if ( *a1 <= 0x1Cu )
    return 0;
  v3 = (unsigned int)v2 - 29;
  if ( v2 != 44 )
  {
    if ( v2 == 63 )
    {
      v3 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      if ( (_DWORD)v3 != 2 )
        return 0;
    }
    else if ( v2 != 42 )
    {
      return 0;
    }
  }
  if ( (a1[7] & 0x40) != 0 )
  {
    v6 = (__int64 *)*((_QWORD *)a1 - 1);
  }
  else
  {
    v3 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v6 = (__int64 *)&a1[-v3];
  }
  if ( *(_BYTE *)*v6 == 84 )
  {
    v3 = *(_QWORD *)(a2 + 32);
    v8 = *(_QWORD *)(*v6 + 40);
    if ( *(_QWORD *)v3 == v8 )
    {
      v4 = *v6;
      if ( (unsigned __int8)sub_D48480(a2, v6[4], v3, v8) )
        return v4;
      return 0;
    }
  }
  if ( v2 == 63 )
    return 0;
  v4 = v6[4];
  if ( *(_BYTE *)v4 != 84 )
    return 0;
  v7 = *(_QWORD *)(v4 + 40);
  if ( **(_QWORD **)(a2 + 32) != v7 || !(unsigned __int8)sub_D48480(a2, *v6, v3, v7) )
    return 0;
  return v4;
}
