// Function: sub_277AC50
// Address: 0x277ac50
//
__int64 __fastcall sub_277AC50(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int8 v4; // bl
  int v5; // r15d
  unsigned int v6; // ebx
  __int64 v7; // r14
  __int64 v8; // r12
  int v9; // r13d
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int8 *v12; // rbx
  __int64 v13; // r14
  unsigned __int8 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rbx
  int v22; // r12d
  int v23; // [rsp+8h] [rbp-68h] BYREF
  int v24; // [rsp+Ch] [rbp-64h] BYREF
  _BYTE *v25; // [rsp+10h] [rbp-60h] BYREF
  _BYTE *v26; // [rsp+18h] [rbp-58h] BYREF
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h] BYREF
  __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v30[7]; // [rsp+38h] [rbp-38h] BYREF

  LOBYTE(v2) = a2 == -4096 || a1 == -8192 || a1 == -4096 || a2 == -8192;
  if ( (_BYTE)v2 )
  {
    LOBYTE(v2) = a1 == a2;
    return v2;
  }
  v4 = *(_BYTE *)a1;
  if ( *(_BYTE *)a2 != *(_BYTE *)a1 )
    return v2;
  if ( (unsigned __int8)sub_B46130(a1, a2, 1) )
  {
    if ( v4 == 85 && ((unsigned __int8)sub_A73ED0((_QWORD *)(a1 + 72), 6) || (unsigned __int8)sub_B49560(a1, 6)) )
    {
      LOBYTE(v2) = *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 40);
      return v2;
    }
    return 1;
  }
  if ( (unsigned int)v4 - 42 <= 0x11 )
  {
    if ( sub_B46D50((unsigned __int8 *)a1) && *(_QWORD *)(a2 - 32) == *(_QWORD *)(a1 - 64) )
      LOBYTE(v2) = *(_QWORD *)(a2 - 64) == *(_QWORD *)(a1 - 32);
    return v2;
  }
  if ( (unsigned __int8)(v4 - 82) <= 1u )
  {
    if ( *(_QWORD *)(a2 - 32) == *(_QWORD *)(a1 - 64) && *(_QWORD *)(a2 - 64) == *(_QWORD *)(a1 - 32) )
      LOBYTE(v2) = (*(_WORD *)(a2 + 2) & 0x3F) == (unsigned int)sub_B52F50(*(_WORD *)(a1 + 2) & 0x3F);
    return v2;
  }
  if ( sub_988010(a1) && sub_988010(a2) )
  {
    v5 = sub_987FE0(a1);
    if ( v5 == (unsigned int)sub_987FE0(a2) )
    {
      if ( sub_277ABC0(a1) )
      {
        if ( (unsigned int)sub_A17190((unsigned __int8 *)a1) > 1 )
        {
          v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
          v11 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
          if ( *(_QWORD *)(a2 + 32 * (1 - v11)) != *(_QWORD *)(a1 - 32 * v10) )
            return v2;
          if ( *(_QWORD *)(a2 - 32 * v11) != *(_QWORD *)(a1 + 32 * (1 - v10)) )
            return v2;
          v12 = sub_24E54B0((unsigned __int8 *)a2);
          v13 = 64 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) + a2;
          v14 = sub_24E54B0((unsigned __int8 *)a1);
          v15 = 64 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) + a1;
          if ( &v14[-v15] != &v12[-v13] )
            return v2;
          v16 = 0;
          while ( v14 != (unsigned __int8 *)(v15 + v16) )
          {
            v17 = *(_QWORD *)(v15 + v16);
            v16 += 32;
            if ( v17 != *(_QWORD *)(v13 + v16 - 32) )
              return v2;
          }
          return 1;
        }
        v4 = *(_BYTE *)a1;
      }
    }
  }
  if ( v4 == 85
    && (v18 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v18
    && *(_QWORD *)(v18 + 24) == *(_QWORD *)(a1 + 80)
    && (*(_BYTE *)(v18 + 33) & 0x20) != 0
    && *(_DWORD *)(v18 + 36) == 149
    && *(_BYTE *)a2 == 85
    && (v19 = *(_QWORD *)(a2 - 32)) != 0
    && !*(_BYTE *)v19
    && *(_QWORD *)(v19 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v19 + 33) & 0x20) != 0
    && *(_DWORD *)(v19 + 36) == 149 )
  {
    if ( *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) == *(_QWORD *)(a1
                                                                                 - 32LL
                                                                                 * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) )
    {
      v20 = sub_B5B740(a1);
      if ( v20 == sub_B5B740(a2) )
      {
        v21 = sub_B5B890(a1);
        LOBYTE(v2) = v21 == sub_B5B890(a2);
      }
    }
  }
  else
  {
    if ( !(unsigned __int8)sub_2779B30(a1, (__int64 *)&v25, &v27, &v29, &v23) )
      return v2;
    v6 = sub_2779B30(a2, (__int64 *)&v26, &v28, v30, &v24);
    if ( !(_BYTE)v6 )
      return v2;
    if ( v23 == v24 )
    {
      if ( (unsigned int)(v23 - 1) <= 3 )
      {
        if ( (v27 != v28 || v29 != v30[0]) && (v27 != v30[0] || v28 != v29) )
          return v2;
        return 1;
      }
      if ( v25 == v26 && v27 == v28 && v29 == v30[0] )
        return 1;
    }
    if ( v27 == v30[0] && v29 == v28 )
    {
      if ( (unsigned __int8)(*v25 - 82) <= 1u
        && (v7 = *((_QWORD *)v25 - 8)) != 0
        && (v8 = *((_QWORD *)v25 - 4)) != 0
        && (v9 = sub_B53900((__int64)v25), (unsigned __int8)(*v26 - 82) <= 1u)
        && v7 == *((_QWORD *)v26 - 8)
        && v8 == *((_QWORD *)v26 - 4)
        && (v22 = sub_B53900((__int64)v26), (unsigned int)sub_B52870(v9) == v22) )
      {
        return v6;
      }
      else
      {
        return 0;
      }
    }
  }
  return v2;
}
