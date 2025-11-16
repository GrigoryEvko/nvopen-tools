// Function: sub_85F680
// Address: 0x85f680
//
__int64 *__fastcall sub_85F680(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r10
  __int64 v6; // r14
  __int64 v9; // rsi
  __int64 *v10; // r15
  char v11; // al
  int v12; // r12d
  __int64 v13; // r9
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-38h]

  v5 = a3;
  v6 = a1;
  if ( *(_BYTE *)(a2 + 28) == 17 && a4 )
  {
    if ( (*(_BYTE *)(a4 + 89) & 2) != 0 )
    {
      v17 = a4;
      v15 = sub_72F070(a4);
      a4 = v17;
      v5 = a3;
      v9 = v15;
    }
    else
    {
      v9 = *(_QWORD *)(a4 + 40);
    }
    v10 = (__int64 *)a4;
    if ( (*(_BYTE *)(a4 + 195) & 3) == 1 )
    {
      v13 = *(_QWORD *)(a4 + 240);
      if ( v13 )
      {
        v11 = *(_BYTE *)(v9 + 28);
        if ( !v11 )
          goto LABEL_10;
        a1 = *(_QWORD *)(a1 + 24);
        v12 = 1;
        goto LABEL_6;
      }
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 16);
    v10 = 0;
  }
  v11 = *(_BYTE *)(v9 + 28);
  if ( !v11 )
    return sub_85C120(
             *(unsigned __int8 *)(a2 + 28),
             *(_DWORD *)(a2 + 24),
             0,
             (__int64)v10,
             0,
             0,
             0,
             0,
             0,
             0,
             (__int64 *)a2,
             0,
             8u);
  v12 = 0;
LABEL_6:
  if ( v11 == 6 )
    sub_85F1C0(a1, v9, *(_QWORD *)(v9 + 32), v5, 0, a4, a5);
  else
    sub_85F1C0(a1, v9, 0, v5, 0, a4, a5);
  if ( v12 )
  {
    v13 = v10[30];
LABEL_10:
    sub_85E1C0(v6, 0, (__int64)v10, *v10, *(_QWORD *)(*(_QWORD *)(*v10 + 96) + 32LL), v13, a5);
  }
  return sub_85C120(
           *(unsigned __int8 *)(a2 + 28),
           *(_DWORD *)(a2 + 24),
           0,
           (__int64)v10,
           0,
           0,
           0,
           0,
           0,
           0,
           (__int64 *)a2,
           0,
           8u);
}
