// Function: sub_2DB29F0
// Address: 0x2db29f0
//
bool __fastcall sub_2DB29F0(__int64 a1, __int64 a2, int a3, int a4)
{
  bool result; // al
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // ebx
  int v16; // eax
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // [rsp-58h] [rbp-58h]
  __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-48h] [rbp-48h]
  __int64 v23; // [rsp-40h] [rbp-40h]

  result = 1;
  if ( a4 != a3 )
  {
    if ( a3 >= 0 )
      return 0;
    if ( a4 >= 0 )
      return 0;
    v7 = sub_2EBEE90(a1, (unsigned int)a3);
    v8 = sub_2EBEE90(a1, (unsigned int)a4);
    v23 = v8;
    if ( !v7 || !v8 || (unsigned __int8)sub_2E8B090(v7) )
      return 0;
    if ( (unsigned int)*(unsigned __int16 *)(v7 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v7 + 32) + 64LL) & 8) != 0
      || ((v9 = *(_DWORD *)(v7 + 44), (v9 & 4) != 0) || (v9 & 8) == 0
        ? (v10 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) >> 19) & 1LL)
        : (LOBYTE(v10) = sub_2E88A90(v7, 0x80000, 1)),
          (_BYTE)v10
       || (unsigned int)*(unsigned __int16 *)(v7 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v7 + 32) + 64LL) & 0x10) != 0
       || ((v18 = *(_DWORD *)(v7 + 44), (v18 & 4) != 0) || (v18 & 8) == 0
         ? (v19 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) >> 20) & 1LL)
         : (LOBYTE(v19) = sub_2E88A90(v7, 0x100000, 1)),
           (_BYTE)v19)) )
    {
      if ( !(unsigned __int8)sub_2E8AED0(v7) )
        return 0;
    }
    v21 = *(_QWORD *)(v7 + 32);
    v20 = 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
    v22 = v21 + v20;
    v11 = 40LL * (unsigned int)sub_2E88FE0(v7);
    v12 = v21 + v11;
    v13 = (__int64)(0xCCCCCCCCCCCCCCCDLL * ((v20 - v11) >> 3)) >> 2;
    if ( v13 > 0 )
    {
      v14 = v12 + 160 * v13;
      while ( *(_BYTE *)v12 || (unsigned int)(*(_DWORD *)(v12 + 8) - 1) > 0x3FFFFFFE )
      {
        if ( !*(_BYTE *)(v12 + 40) && (unsigned int)(*(_DWORD *)(v12 + 48) - 1) <= 0x3FFFFFFE )
        {
          v12 += 40;
          break;
        }
        if ( !*(_BYTE *)(v12 + 80) && (unsigned int)(*(_DWORD *)(v12 + 88) - 1) <= 0x3FFFFFFE )
        {
          v12 += 80;
          break;
        }
        if ( !*(_BYTE *)(v12 + 120) && (unsigned int)(*(_DWORD *)(v12 + 128) - 1) <= 0x3FFFFFFE )
        {
          v12 += 120;
          break;
        }
        v12 += 160;
        if ( v14 == v12 )
          goto LABEL_36;
      }
LABEL_21:
      if ( v22 == v12 )
        goto LABEL_22;
      return 0;
    }
LABEL_36:
    v17 = v22 - v12;
    if ( v22 - v12 != 80 )
    {
      if ( v17 != 120 )
      {
        if ( v17 != 40 )
          goto LABEL_22;
        goto LABEL_39;
      }
      if ( !*(_BYTE *)v12 && (unsigned int)(*(_DWORD *)(v12 + 8) - 1) <= 0x3FFFFFFE )
        goto LABEL_21;
      v12 += 40;
    }
    if ( !*(_BYTE *)v12 && (unsigned int)(*(_DWORD *)(v12 + 8) - 1) <= 0x3FFFFFFE )
      goto LABEL_21;
    v12 += 40;
LABEL_39:
    if ( *(_BYTE *)v12 || (unsigned int)(*(_DWORD *)(v12 + 8) - 1) > 0x3FFFFFFE )
    {
LABEL_22:
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a2 + 280LL))(
             a2,
             v7,
             v23,
             a1) )
      {
        v15 = sub_2E8E710(v7, (unsigned int)a3, 0, 0, 0);
        v16 = sub_2E8E710(v23, (unsigned int)a4, 0, 0, 0);
        if ( v15 != -1 && v16 != -1 )
          return v15 == v16;
      }
      return 0;
    }
    goto LABEL_21;
  }
  return result;
}
