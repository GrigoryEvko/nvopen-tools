// Function: sub_25BDC30
// Address: 0x25bdc30
//
__int64 __fastcall sub_25BDC30(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // r12
  unsigned int v5; // r8d
  __int64 v6; // rax
  int v7; // edx
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdi
  __int64 v14; // r9
  int v15; // edi
  int v16; // r11d
  __int64 v17; // [rsp-20h] [rbp-20h] BYREF

  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    return 0;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)*a2 - 34) )
    return 0;
  v4 = *a1;
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 29) )
    return 0;
  v5 = sub_B49560((__int64)a2, 29);
  if ( (_BYTE)v5 )
    return 0;
  v6 = *((_QWORD *)a2 - 4);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a2 + 10) )
    return 1;
  v7 = *(_DWORD *)(v4 + 16);
  v17 = *((_QWORD *)a2 - 4);
  if ( !v7 )
  {
    v8 = *(_QWORD **)(v4 + 32);
    v9 = &v8[*(unsigned int *)(v4 + 40)];
    if ( v9 == sub_25BD100(v8, (__int64)v9, &v17) )
      return 1;
    return v5;
  }
  v10 = *(unsigned int *)(v4 + 24);
  v11 = *(_QWORD *)(v4 + 8);
  if ( (_DWORD)v10 )
  {
    v12 = (v10 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v13 = (__int64 *)(v11 + 8LL * v12);
    v14 = *v13;
    if ( v6 != *v13 )
    {
      v15 = 1;
      while ( v14 != -4096 )
      {
        v16 = v15 + 1;
        v12 = (v10 - 1) & (v15 + v12);
        v13 = (__int64 *)(v11 + 8LL * v12);
        v14 = *v13;
        if ( v6 == *v13 )
          goto LABEL_17;
        v15 = v16;
      }
      return 1;
    }
LABEL_17:
    if ( v13 != (__int64 *)(v11 + 8 * v10) )
      return v5;
  }
  return 1;
}
