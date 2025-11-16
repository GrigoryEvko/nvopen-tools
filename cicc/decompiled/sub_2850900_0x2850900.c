// Function: sub_2850900
// Address: 0x2850900
//
__int64 __fastcall sub_2850900(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        unsigned __int8 a5,
        unsigned int a6,
        _QWORD **a7,
        __int64 a8)
{
  __int64 v10; // rdx
  unsigned int v12; // esi
  __int64 v13; // r11
  char v14; // r10
  __int64 *v16; // rdi
  char v17; // r9
  char v18; // al
  __int64 v19; // rax
  char v20; // r9
  __int64 v21; // rax
  __int64 v22; // [rsp+10h] [rbp-50h]
  char v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  char v28; // [rsp+28h] [rbp-38h]
  unsigned __int8 v29; // [rsp+28h] [rbp-38h]
  char v30; // [rsp+2Fh] [rbp-31h]

  v10 = (__int64)a7;
  if ( *(_DWORD *)(a2 + 32) != a6 )
    return 0;
  v12 = a6;
  v13 = *(_QWORD *)(a2 + 712);
  v14 = *(_BYTE *)(a2 + 720);
  v27 = *(_QWORD *)(a2 + 728);
  v30 = *(_BYTE *)(a2 + 736);
  if ( a6 == 2 && a7 != *(_QWORD ***)(a2 + 40) )
  {
    v29 = a5;
    v23 = *(_BYTE *)(a2 + 720);
    v26 = *(_QWORD *)(a2 + 712);
    v21 = sub_BCB120(*a7);
    v12 = 2;
    a5 = v29;
    v14 = v23;
    v13 = v26;
    v10 = v21;
  }
  if ( a4 )
  {
    if ( !*(_BYTE *)(a2 + 720) || a3 >= *(_QWORD *)(a2 + 712) )
    {
LABEL_15:
      if ( *(_QWORD *)(a2 + 728) < a3 )
      {
        v19 = *(_QWORD *)(a2 + 712);
        v20 = a4;
        if ( v19 )
          v20 = *(_BYTE *)(a2 + 720);
        v25 = v10;
        v28 = v14;
        v22 = v13;
        if ( !sub_2850840(*(__int64 **)(a1 + 48), v12, v10, (unsigned int)a8, a3 - v19, v20, a5) )
          return 0;
        v14 = v28;
        v13 = v22;
        v30 = a4;
        v27 = a3;
        v10 = v25;
      }
      goto LABEL_6;
    }
LABEL_11:
    v16 = *(__int64 **)(a1 + 48);
    v17 = *(_BYTE *)(a2 + 736);
    v24 = v10;
    if ( a3 )
      v17 = a4;
    v18 = sub_2850840(v16, v12, v10, (unsigned int)a8, *(_QWORD *)(a2 + 728) - a3, v17, a5);
    v10 = v24;
    v14 = a4;
    v13 = a3;
    if ( !v18 )
      return 0;
    goto LABEL_6;
  }
  if ( a3 < *(_QWORD *)(a2 + 712) )
    goto LABEL_11;
  if ( !*(_BYTE *)(a2 + 736) )
    goto LABEL_15;
LABEL_6:
  if ( !v10 || *(_BYTE *)(v10 + 8) != 7 || !v14 && !v30 )
  {
    *(_DWORD *)(a2 + 48) = a8;
    *(_QWORD *)(a2 + 712) = v13;
    *(_QWORD *)(a2 + 728) = v27;
    *(_BYTE *)(a2 + 720) = v14;
    *(_BYTE *)(a2 + 736) = v30;
    *(_QWORD *)(a2 + 40) = v10;
    return 1;
  }
  return 0;
}
