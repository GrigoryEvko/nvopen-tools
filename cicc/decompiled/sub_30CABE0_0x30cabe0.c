// Function: sub_30CABE0
// Address: 0x30cabe0
//
__int64 __fastcall sub_30CABE0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, char a5)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 result; // rax

  *(_QWORD *)a1 = &unk_4A1F3E0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = sub_B491C0((__int64)a3);
  v8 = *(a3 - 4);
  if ( v8 )
  {
    if ( *(_BYTE *)v8 )
    {
      v8 = 0;
    }
    else if ( *(_QWORD *)(v8 + 24) != a3[10] )
    {
      v8 = 0;
    }
  }
  *(_QWORD *)(a1 + 24) = v8;
  v9 = a3[6];
  *(_QWORD *)(a1 + 32) = v9;
  if ( v9 )
    sub_B96E90(a1 + 32, v9, 1);
  result = a3[5];
  *(_QWORD *)(a1 + 48) = a4;
  *(_BYTE *)(a1 + 56) = a5;
  *(_QWORD *)(a1 + 40) = result;
  *(_BYTE *)(a1 + 57) = 0;
  return result;
}
