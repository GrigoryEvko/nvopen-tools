// Function: sub_B4CED0
// Address: 0xb4ced0
//
__int64 __fastcall sub_B4CED0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  char v5; // bl
  __int64 v6; // rax
  char v7; // dl
  char v8; // r13
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  char v13; // dl

  v4 = *(_QWORD *)(a2 + 72);
  v5 = sub_AE5020(a3, v4);
  v6 = sub_9208B0(a3, v4);
  v8 = v7;
  v9 = ((1LL << v5) + ((unsigned __int64)(v6 + 7) >> 3) - 1) >> v5 << v5;
  if ( (unsigned __int8)sub_B4CE70(a2) )
  {
    v10 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v10 != 17 )
      goto LABEL_6;
    v11 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v11 = (_QWORD *)*v11;
    v12 = sub_B48740(v9, (__int64)v11);
    if ( v13 )
    {
      *(_BYTE *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v12;
    }
    else
    {
LABEL_6:
      *(_BYTE *)(a1 + 16) = 0;
    }
  }
  else
  {
    *(_QWORD *)a1 = v9;
    *(_BYTE *)(a1 + 8) = v8;
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
