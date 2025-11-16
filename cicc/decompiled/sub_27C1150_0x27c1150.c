// Function: sub_27C1150
// Address: 0x27c1150
//
__int64 __fastcall sub_27C1150(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v3; // rsi
  unsigned __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int8 v10; // cl
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rsi
  __int64 *v16; // rax

  v3 = (_QWORD *)(a2 + 48);
  v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v5 == v3 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = *(unsigned __int8 *)(v5 - 24);
    v7 = 0;
    v8 = v5 - 24;
    if ( (unsigned int)(v6 - 30) < 0xB )
      v7 = v8;
  }
  v9 = sub_B46EC0(v7, 0);
  v10 = *(_BYTE *)(a1 + 84);
  v11 = v9;
  if ( v10 )
  {
    v12 = *(_QWORD **)(a1 + 64);
    v13 = &v12[*(unsigned int *)(a1 + 76)];
    if ( v12 == v13 )
    {
LABEL_14:
      LOBYTE(v14) = 0;
    }
    else
    {
      while ( v11 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_14;
      }
      LOBYTE(v14) = *(_BYTE *)(a1 + 84);
      v10 = 0;
    }
  }
  else
  {
    v16 = sub_C8CA60(a1 + 56, v9);
    LOBYTE(v14) = v16 != 0;
    v10 = v16 == 0;
  }
  v14 = (unsigned __int8)v14;
  if ( a3 )
    v14 = v10;
  return sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v7 - 96) + 8LL), v14, 0);
}
