// Function: sub_2E8B400
// Address: 0x2e8b400
//
__int64 __fastcall sub_2E8B400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  unsigned int v5; // r13d
  _BYTE *v6; // rbx
  int v7; // eax
  __int64 v8; // rax
  int v9; // eax
  char v10; // al
  int v11; // eax
  int v13; // eax
  __int64 v14; // rax
  __int16 v15; // ax
  int v16; // eax
  __int64 v17; // rax

  v6 = (_BYTE *)a2;
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) != 0 )
    goto LABEL_10;
  v7 = *(_DWORD *)(a1 + 44);
  if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
  {
    v8 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 20) & 1LL;
  }
  else
  {
    a2 = 0x100000;
    LOBYTE(v8) = sub_2E88A90(a1, 0x100000, 1);
  }
  if ( (_BYTE)v8 )
    goto LABEL_10;
  v9 = *(_DWORD *)(a1 + 44);
  if ( (v9 & 4) != 0 || (v9 & 8) == 0 )
  {
    v10 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 7;
  }
  else
  {
    a2 = 128;
    v10 = sub_2E88A90(a1, 128, 1);
  }
  if ( v10 )
    goto LABEL_10;
  v11 = *(unsigned __int16 *)(a1 + 68);
  LOBYTE(a3) = v11 == 0;
  LOBYTE(v5) = v11 == 0 || v11 == 68;
  if ( (_BYTE)v5 )
    goto LABEL_10;
  if ( ((unsigned int)(v11 - 1) <= 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) != 0
     || ((v13 = *(_DWORD *)(a1 + 44), (v13 & 4) != 0) || (v13 & 8) == 0
       ? (v14 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 19) & 1LL)
       : (a2 = 0x80000, LOBYTE(v14) = sub_2E88A90(a1, 0x80000, 1)),
         (_BYTE)v14))
    && (unsigned __int8)sub_2E8B100(a1, a2, a3, a4, a5) )
  {
LABEL_10:
    *v6 = 1;
    return 0;
  }
  else
  {
    v15 = *(_WORD *)(a1 + 68);
    if ( (unsigned __int16)(v15 - 3) > 3u
      && (unsigned __int16)(v15 - 14) > 4u
      && !(unsigned __int8)sub_2E50190(a1, 9, 1)
      && *(_WORD *)(a1 + 68) != 45
      && (!(unsigned __int8)sub_2E50190(a1, 21, 1) || (*(_BYTE *)(a1 + 45) & 0x40) != 0)
      && !(unsigned __int8)sub_2E50190(a1, 24, 1)
      && (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 )
    {
      v16 = *(_DWORD *)(a1 + 44);
      if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
        v17 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 19) & 1LL;
      else
        LOBYTE(v17) = sub_2E88A90(a1, 0x80000, 1);
      if ( !(_BYTE)v17 || (unsigned __int8)sub_2E8AED0(a1) )
        return 1;
      else
        return (unsigned __int8)*v6 ^ 1u;
    }
  }
  return v5;
}
