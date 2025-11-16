// Function: sub_5C7300
// Address: 0x5c7300
//
__int64 __fastcall sub_5C7300(__int64 a1, __int64 a2)
{
  char v3; // al
  _QWORD *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax

  v3 = *(_BYTE *)(a2 + 200);
  v4 = (_QWORD *)(a1 + 56);
  if ( v3 < 0 )
  {
    sub_6851C0(2537, v4);
    return a2;
  }
  else
  {
    if ( (v3 & 0x20) != 0 )
    {
      sub_6851C0(2538, v4);
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 32);
      v6 = *(_QWORD *)a2;
      v7 = *(_QWORD *)(v5 + 40);
      *(_BYTE *)(a2 + 201) |= 1u;
      sub_5C7230(v6, 0, *(_QWORD *)(v7 + 184), v4);
    }
    return a2;
  }
}
