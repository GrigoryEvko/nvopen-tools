// Function: sub_80A250
// Address: 0x80a250
//
__int64 __fastcall sub_80A250(__int64 a1, char a2, char a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 result; // rax
  __int64 v8; // rsi
  char v9; // dl
  __int64 *v10; // rdx

  v6 = a1;
  if ( a2 == 6 )
  {
    v6 = sub_809390(a1);
    result = qword_4F18BC8;
    if ( qword_4F18BC8 )
      goto LABEL_3;
  }
  else
  {
    result = qword_4F18BC8;
    if ( qword_4F18BC8 )
    {
LABEL_3:
      qword_4F18BC8 = *(_QWORD *)result;
      goto LABEL_4;
    }
  }
  result = sub_822B10(48);
LABEL_4:
  *(_BYTE *)(result + 24) = a2;
  v8 = qword_4F18C00[BYTE1(v6)];
  *(_QWORD *)(result + 16) = v6;
  qword_4F18C00[BYTE1(v6)] = result;
  *(_QWORD *)(result + 8) = v8;
  *(_BYTE *)(v6 + 91) |= 2u;
  v9 = *(_BYTE *)(result + 40);
  *(_QWORD *)result = 0;
  *(_BYTE *)(result + 40) = a3 & 1 | v9 & 0xFE;
  v10 = *(__int64 **)(a4 + 24);
  if ( v10 )
  {
    *(_QWORD *)(result + 32) = v10[4] + 1;
    *v10 = result;
    *(_QWORD *)(a4 + 24) = result;
  }
  else
  {
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(a4 + 24) = result;
    *(_QWORD *)(a4 + 16) = result;
  }
  return result;
}
