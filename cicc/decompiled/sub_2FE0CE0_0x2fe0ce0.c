// Function: sub_2FE0CE0
// Address: 0x2fe0ce0
//
__int64 __fastcall sub_2FE0CE0(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  __int64 v7; // rbx
  unsigned __int16 v8; // ax
  unsigned int v9; // r8d
  int v11; // eax
  __int64 v12; // rax
  __int64 (*v13)(); // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  unsigned __int8 *v16; // rcx
  unsigned __int64 v17; // rdx

  v7 = *a3;
  v8 = *(_WORD *)(*a3 + 68);
  if ( v8 == 3 )
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 1384LL))(a1, a2, a3, a4);
  if ( (unsigned __int16)(v8 - 4) <= 2u || (unsigned int)v8 - 1 <= 1 )
    return 2;
  v9 = 3;
  if ( (unsigned __int16)(v8 - 14) <= 4u || v8 <= 0x17u && ((1LL << v8) & 0xC00480) != 0 )
    return v9;
  v11 = *(_DWORD *)(v7 + 44);
  if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
    v12 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) >> 9) & 1LL;
  else
    LOBYTE(v12) = sub_2E88A90(v7, 512, 1);
  if ( (_BYTE)v12 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(v7 + 24) + 120LL) )
      return 2;
    v13 = *(__int64 (**)())(*(_QWORD *)a1 + 920LL);
    if ( v13 != sub_2DB1B30 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v13)(a1, v7) )
        return 2;
    }
  }
  v14 = *(unsigned __int8 **)(v7 + 32);
  v15 = 2384;
  v16 = &v14[40 * (*(_DWORD *)(v7 + 40) & 0xFFFFFF)];
  if ( v14 != v16 )
  {
    while ( 1 )
    {
      v17 = *v14;
      if ( (unsigned __int8)v17 <= 0xBu )
      {
        if ( _bittest64(&v15, v17) )
          break;
      }
      v14 += 40;
      if ( v16 == v14 )
        return (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 1384LL))(a1, a2, a3, a4);
    }
    return 2;
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 1384LL))(a1, a2, a3, a4);
}
