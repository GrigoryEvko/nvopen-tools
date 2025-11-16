// Function: sub_2E980E0
// Address: 0x2e980e0
//
__int64 __fastcall sub_2E980E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  _BYTE *v4; // r13
  _BYTE *v5; // rbx
  _BYTE *v6; // r12
  _BYTE *v7; // rbx

  if ( *(_WORD *)(a2 + 68) != 10 || (v2 = *(_DWORD *)(a2 + 40) & 0xFFFFFF, (_DWORD)v2 != 1) )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 27LL) & 0x20) == 0
      || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 56LL))(a1, a2) )
    {
      return 0;
    }
    v2 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  }
  v3 = *(_QWORD *)(a2 + 32);
  v4 = (_BYTE *)(v3 + 40 * v2);
  v5 = (_BYTE *)(v3 + 40LL * (unsigned int)sub_2E88FE0(a2));
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      v6 = v5;
      if ( (unsigned __int8)sub_2E2FA70(v5) )
        break;
      v5 += 40;
      if ( v4 == v5 )
        return 1;
    }
    while ( v4 != v6 )
    {
      if ( *((int *)v6 + 2) < 0 )
        return 0;
      v7 = v6 + 40;
      if ( v6 + 40 == v4 )
        break;
      while ( 1 )
      {
        v6 = v7;
        if ( (unsigned __int8)sub_2E2FA70(v7) )
          break;
        v7 += 40;
        if ( v4 == v7 )
          return 1;
      }
    }
  }
  return 1;
}
