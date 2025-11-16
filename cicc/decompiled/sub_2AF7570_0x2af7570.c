// Function: sub_2AF7570
// Address: 0x2af7570
//
__int64 __fastcall sub_2AF7570(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v4; // r9
  _QWORD *v7; // rsi
  int v8; // r15d
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(*a2 + 40LL);
  v17 = *a2 + 24LL;
  v4 = *(_QWORD *)(v3 + 56);
  if ( v4 == v3 + 48 )
  {
    v12 = *a2 + 24LL;
  }
  else
  {
    v7 = &a2[a3];
    v8 = 0;
    while ( 1 )
    {
      v9 = v4 - 24;
      if ( !v4 )
        v9 = 0;
      v18[0] = v9;
      if ( v7 != sub_2AF7070(a2, (__int64)v7, v18) )
      {
        v14 = v10 + 24;
        if ( ++v8 == 1 )
          v12 = v14;
        if ( v8 == a3 )
          break;
      }
      v4 = *(_QWORD *)(v11 + 8);
      if ( v13 == v4 )
        goto LABEL_10;
    }
    v17 = v14;
  }
LABEL_10:
  v15 = *(_QWORD *)(v17 + 8);
  *(_QWORD *)a1 = v12;
  *(_WORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = v15;
  *(_WORD *)(a1 + 24) = 0;
  return a1;
}
