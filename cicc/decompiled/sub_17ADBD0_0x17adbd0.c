// Function: sub_17ADBD0
// Address: 0x17adbd0
//
__int64 __fastcall sub_17ADBD0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  int v7; // edx
  __int64 v8; // rbx
  _BYTE *v9; // r14
  unsigned __int8 v10; // al
  __int64 v11; // rsi
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  v7 = *(unsigned __int8 *)(a2 + 16);
  v8 = *(_QWORD *)(a2 - 48);
  v9 = *(_BYTE **)(a2 - 24);
  if ( v7 != 47 )
  {
    if ( v7 == 51 )
    {
      v10 = v9[16];
      v11 = (__int64)(v9 + 24);
      if ( v10 != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
          goto LABEL_8;
        if ( v10 > 0x10u )
          goto LABEL_8;
        v15 = sub_15A1020(v9, v11, *(_QWORD *)v9, (unsigned int)(v7 - 24));
        if ( !v15 || *(_BYTE *)(v15 + 16) != 13 )
          goto LABEL_8;
        v11 = v15 + 24;
      }
      if ( (unsigned __int8)sub_14C1670(v8, v11, a3, 0, 0, 0, 0) )
      {
        *(_DWORD *)a1 = 11;
        *(_QWORD *)(a1 + 8) = v8;
        *(_QWORD *)(a1 + 16) = v9;
        return a1;
      }
    }
LABEL_8:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return a1;
  }
  if ( v9[16] > 0x10u )
    goto LABEL_8;
  v13 = (__int64 *)sub_15A0680(*(_QWORD *)a2, 1, 0);
  v14 = sub_15A2D50(v13, (__int64)v9, 0, 0, a4, a5, a6);
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = v14;
  *(_DWORD *)a1 = 15;
  return a1;
}
