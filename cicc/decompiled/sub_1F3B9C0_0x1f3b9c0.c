// Function: sub_1F3B9C0
// Address: 0x1f3b9c0
//
__int64 __fastcall sub_1F3B9C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  int v8; // r15d
  __int64 v9; // rax
  __int64 (*v10)(); // rax
  __int16 v11; // ax
  __int64 v12; // rax
  __int64 v14; // rax
  __int16 v15; // dx
  bool v16; // al
  __int64 v17; // rax
  __int16 v18; // dx
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rax
  signed int v24; // esi
  char v25; // al
  __int64 v26; // rdx
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  _DWORD v28[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = sub_1E15F70(a2);
  if ( !*(_DWORD *)(a2 + 40) )
    return 0;
  v6 = v5;
  v7 = *(_QWORD *)(a2 + 32);
  if ( *(_BYTE *)v7 )
    return 0;
  v8 = *(_DWORD *)(v7 + 8);
  v27 = *(_QWORD **)(v6 + 40);
  if ( v8 < 0 && (*(_DWORD *)v7 & 0xFFF00) != 0 && (unsigned __int8)sub_1E166B0(a2, v8, 0) )
    return 0;
  v9 = *a1;
  v28[0] = 0;
  v10 = *(__int64 (**)())(v9 + 48);
  if ( v10 == sub_1E1C810
    || !((unsigned int (__fastcall *)(__int64 *, __int64, _DWORD *))v10)(a1, a2, v28)
    || (v26 = *(_QWORD *)(v6 + 56), *(_BYTE *)(v26 + 657))
    || !*(_BYTE *)(*(_QWORD *)(v26 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v26 + 32) + v28[0]) + 20) )
  {
    v11 = *(_WORD *)(a2 + 46);
    if ( (v11 & 4) == 0 && (v11 & 8) != 0 )
      LOBYTE(v12) = sub_1E15D00(a2, 0x80000u, 1);
    else
      v12 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 19) & 1LL;
    if ( (_BYTE)v12 )
      return 0;
    v14 = *(_QWORD *)(a2 + 16);
    if ( *(_WORD *)v14 == 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) != 0 )
      return 0;
    v15 = *(_WORD *)(a2 + 46);
    if ( (v15 & 4) == 0 && (v15 & 8) != 0 )
      v16 = sub_1E15D00(a2, 0x20000u, 1);
    else
      v16 = (*(_QWORD *)(v14 + 8) & 0x20000LL) != 0;
    if ( v16 )
      return 0;
    if ( sub_1E17880(a2) )
      return 0;
    v17 = *(_QWORD *)(a2 + 16);
    if ( *(_WORD *)v17 == 1 )
      return 0;
    v18 = *(_WORD *)(a2 + 46);
    if ( (v18 & 4) != 0 || (v18 & 8) == 0 )
      v19 = WORD1(*(_QWORD *)(v17 + 8)) & 1;
    else
      v19 = sub_1E15D00(a2, 0x10000u, 1);
    if ( v19 && !(unsigned __int8)sub_1E176D0(a2, a3) )
      return 0;
    v20 = *(unsigned int *)(a2 + 40);
    if ( (_DWORD)v20 )
    {
      v21 = 0;
      v22 = 40 * v20;
      do
      {
        v23 = v21 + *(_QWORD *)(a2 + 32);
        if ( !*(_BYTE *)v23 )
        {
          v24 = *(_DWORD *)(v23 + 8);
          if ( v24 )
          {
            v25 = *(_BYTE *)(v23 + 3) & 0x10;
            if ( v24 > 0 )
            {
              if ( v25 || !(unsigned __int8)sub_1E69FD0(v27, v24) )
                return 0;
            }
            else if ( v8 != v24 || !v25 )
            {
              return 0;
            }
          }
        }
        v21 += 40;
      }
      while ( v21 != v22 );
    }
  }
  return 1;
}
