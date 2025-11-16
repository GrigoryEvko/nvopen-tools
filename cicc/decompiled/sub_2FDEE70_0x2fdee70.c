// Function: sub_2FDEE70
// Address: 0x2fdee70
//
__int64 __fastcall sub_2FDEE70(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rax
  signed int v5; // r15d
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 (*v8)(); // rax
  __int64 v9; // rax
  int *v10; // rdx
  int v11; // eax
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 i; // r13
  unsigned int v25; // esi
  char v26; // al
  _DWORD v27[13]; // [rsp+Ch] [rbp-34h] BYREF

  v2 = sub_2E88D60(a2);
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) == 0 )
    return 0;
  v3 = v2;
  v4 = *(_QWORD *)(a2 + 32);
  if ( *(_BYTE *)v4 )
    return 0;
  v5 = *(_DWORD *)(v4 + 8);
  v6 = *(_QWORD **)(v3 + 32);
  if ( v5 < 0 && (*(_DWORD *)v4 & 0xFFF00) != 0 && (unsigned __int8)sub_2E89D80(a2, v5, 0) )
    return 0;
  v7 = *a1;
  v27[0] = 0;
  v8 = *(__int64 (**)())(v7 + 88);
  if ( v8 == sub_2E97330
    || !((unsigned int (__fastcall *)(__int64 *, __int64, _DWORD *))v8)(a1, a2, v27)
    || (v13 = *(_QWORD *)(v3 + 48), *(_BYTE *)(v13 + 670))
    || (result = *(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v13 + 32) + v27[0]) + 17),
        !(_BYTE)result) )
  {
    v9 = *(_QWORD *)(a2 + 48);
    v10 = (int *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v11 = v9 & 7;
      switch ( v11 )
      {
        case 1:
          return 0;
        case 3:
          v14 = *((unsigned __int8 *)v10 + 4);
          if ( (_BYTE)v14 && *(_QWORD *)&v10[2 * *v10 + 4]
            || *((_BYTE *)v10 + 5) && *(_QWORD *)&v10[2 * *v10 + 4 + 2 * v14] )
          {
            return 0;
          }
          break;
        case 2:
          return 0;
      }
    }
    v15 = *(_DWORD *)(a2 + 44);
    if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
      v16 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 23) & 1LL;
    else
      LOBYTE(v16) = sub_2E88A90(a2, 0x800000, 1);
    if ( (_BYTE)v16
      || (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) != 0 )
    {
      return 0;
    }
    v17 = *(_DWORD *)(a2 + 44);
    if ( (v17 & 4) != 0 || (v17 & 8) == 0 )
      v18 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 20) & 1LL;
    else
      LOBYTE(v18) = sub_2E88A90(a2, 0x100000, 1);
    if ( (_BYTE)v18 )
      return 0;
    v19 = *(_DWORD *)(a2 + 44);
    if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
      v20 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 21) & 1LL;
    else
      LOBYTE(v20) = sub_2E88A90(a2, 0x200000, 1);
    if ( (_BYTE)v20 && (*(_BYTE *)(a2 + 45) & 0x40) == 0
      || sub_2E8B090(a2)
      || (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1 )
    {
      return 0;
    }
    v21 = *(_DWORD *)(a2 + 44);
    if ( (v21 & 4) != 0 || (v21 & 8) == 0 )
      v22 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 19) & 1LL;
    else
      LOBYTE(v22) = sub_2E88A90(a2, 0x80000, 1);
    if ( (_BYTE)v22 && !(unsigned __int8)sub_2E8AED0(a2) )
      return 0;
    v23 = *(_QWORD *)(a2 + 32);
    for ( i = v23 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF); i != v23; v23 += 40 )
    {
      if ( !*(_BYTE *)v23 )
      {
        v25 = *(_DWORD *)(v23 + 8);
        if ( v25 )
        {
          v26 = *(_BYTE *)(v23 + 3) & 0x10;
          if ( v25 - 1 <= 0x3FFFFFFE )
          {
            if ( v26 || !(unsigned __int8)sub_2EBF3A0(v6, v25) )
              return 0;
          }
          else if ( !v26 || v5 != v25 )
          {
            return 0;
          }
        }
      }
    }
    return 1;
  }
  return result;
}
