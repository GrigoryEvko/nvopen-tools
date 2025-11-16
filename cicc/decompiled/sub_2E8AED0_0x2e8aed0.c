// Function: sub_2E8AED0
// Address: 0x2e8aed0
//
__int64 __fastcall sub_2E8AED0(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rax
  int *v4; // rdx
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // r13
  int *v8; // rbx
  int *v9; // r13
  __int64 *v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi

  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) == 0 )
  {
    v1 = *(_DWORD *)(a1 + 44);
    if ( (v1 & 4) != 0 || (v1 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & 0x80000LL) == 0 )
        return 0;
    }
    else if ( !sub_2E88A90(a1, 0x80000, 1) )
    {
      return 0;
    }
  }
  v3 = *(_QWORD *)(a1 + 48);
  v4 = (int *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  if ( (v3 & 7) == 0 )
  {
    *(_QWORD *)(a1 + 48) = v4;
    LOBYTE(v3) = v3 & 0xF8;
    goto LABEL_12;
  }
  if ( (v3 & 7) != 3 || !*v4 )
    return 0;
LABEL_12:
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 48LL);
  v6 = v3 & 7;
  if ( v6 )
  {
    if ( v6 != 3 )
      return 1;
    v8 = v4 + 4;
    v7 = 2LL * *v4;
  }
  else
  {
    *(_QWORD *)(a1 + 48) = v4;
    v7 = 2;
    v8 = (int *)(a1 + 48);
  }
  v9 = &v8[v7];
  if ( v9 != v8 )
  {
    while ( 1 )
    {
      v10 = *(__int64 **)v8;
      if ( (*(_BYTE *)(*(_QWORD *)v8 + 37LL) & 0xFu) > 1 )
        break;
      if ( (v10[4] & 6) != 0 )
        break;
      if ( (v10[4] & 0x30) != 0x30 )
      {
        v11 = *v10;
        if ( !*v10 )
          break;
        if ( (v11 & 4) == 0 )
          break;
        v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v12 || !(*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v12 + 24LL))(v12, v5) )
          break;
      }
      v8 += 2;
      if ( v9 == v8 )
        return 1;
    }
    return 0;
  }
  return 1;
}
