// Function: sub_830AC0
// Address: 0x830ac0
//
_QWORD *__fastcall sub_830AC0(__int64 a1, int a2, int a3)
{
  _QWORD *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE *v10; // r12
  char v11; // al
  _QWORD *result; // rax
  __int64 v13; // rax

  v4 = sub_830A00(a2);
  v5 = *(_QWORD *)(a1 + 24);
  v10 = sub_73DE50((__int64)v4, v5);
  v11 = *(_BYTE *)(a1 + 32);
  if ( (v11 & 3) != 0 )
    goto LABEL_5;
  v6 = *(_QWORD *)(a1 + 8);
  if ( !v6 || (*(_BYTE *)(v6 + 172) & 1) == 0 )
  {
    if ( (v11 & 4) == 0 || (v11 & 8) != 0 )
      goto LABEL_5;
LABEL_11:
    v13 = sub_72D2E0(*(_QWORD **)v10);
    result = sub_73DBF0(0, v13, (__int64)v10);
    *((_BYTE *)result + 27) |= 2u;
    return result;
  }
  if ( (v11 & 8) == 0 )
    goto LABEL_11;
LABEL_5:
  if ( a3 )
    return v10;
  else
    return sub_731370((__int64)v10, v5, v6, v7, v8, v9);
}
