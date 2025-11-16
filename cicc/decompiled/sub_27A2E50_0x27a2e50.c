// Function: sub_27A2E50
// Address: 0x27a2e50
//
__int64 __fastcall sub_27A2E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rbx
  _QWORD *v14; // r14
  __int64 v15; // r15
  __int64 v16; // r15
  __int64 v17; // rax
  _BYTE *v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  v7 = (unsigned __int8 *)sub_B47F80((_BYTE *)a5);
  v22 = 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF);
  v8 = 0;
  if ( (*(_DWORD *)(a5 + 4) & 0x7FFFFFF) != 0 )
  {
    do
    {
      if ( (*(_BYTE *)(a5 + 7) & 0x40) != 0 )
        v9 = *(_QWORD *)(a5 - 8);
      else
        v9 = a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF);
      v10 = *(_QWORD *)(v9 + v8);
      if ( *(_BYTE *)v10 > 0x1Cu )
      {
        v21 = *(_BYTE **)(v9 + v8);
        if ( !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 216), *(_QWORD *)(v10 + 40), a3) && *v21 == 63 )
          sub_27A2E50(a1, v7, a3, a4);
      }
      v8 += 32;
    }
    while ( v22 != v8 );
  }
  v11 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == a3 + 48 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v11 )
      BUG();
    v12 = v11 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 >= 0xB )
      v12 = 0;
  }
  sub_B44220(v7, v12 + 24, 0);
  sub_B9ADA0((__int64)v7, 0, 0);
  v13 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  v14 = *(_QWORD **)a4;
  if ( v13 != *(_QWORD *)a4 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(*v14 - 32LL);
      if ( !v15 )
        break;
      sub_B45560(v7, *(_QWORD *)(*v14 - 32LL));
      if ( a5 == v15 )
      {
        if ( (_QWORD *)v13 == ++v14 )
          return sub_BD2ED0(a2, a5, (__int64)v7);
      }
      else
      {
LABEL_18:
        ++v14;
        v16 = sub_B10CD0(v15 + 48);
        v17 = sub_B10CD0((__int64)(v7 + 48));
        sub_AE8F10((__int64)v7, v17, v16);
        if ( (_QWORD *)v13 == v14 )
          return sub_BD2ED0(a2, a5, (__int64)v7);
      }
    }
    sub_B45560(v7, 0);
    goto LABEL_18;
  }
  return sub_BD2ED0(a2, a5, (__int64)v7);
}
