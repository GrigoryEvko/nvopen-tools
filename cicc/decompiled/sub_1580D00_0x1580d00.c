// Function: sub_1580D00
// Address: 0x1580d00
//
__int64 __fastcall sub_1580D00(__int64 a1, __int64 a2, char a3)
{
  char v3; // al
  __int64 v4; // r13
  __int64 v5; // rax
  char v7; // r13
  unsigned int v8; // ebx
  __int64 v9; // r12
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v3 = *(_BYTE *)(a1 + 8);
    if ( v3 == 14 )
    {
      v4 = sub_15A0680(a2, *(_QWORD *)(a1 + 32), 0);
      v5 = sub_1580D00(*(_QWORD *)(a1 + 24), a2, 1);
      return sub_15A2C20(v5, v4, 1, 0);
    }
    v7 = a3;
    if ( v3 == 13 )
    {
      if ( (*(_BYTE *)(a1 + 9) & 2) != 0 )
        break;
      v8 = *(_DWORD *)(a1 + 12);
      if ( !v8 )
        return sub_15A06D0(a2);
      v16 = sub_1580D00(**(_QWORD **)(a1 + 16), a2, 1);
      if ( v8 == 1 )
      {
LABEL_19:
        v15 = sub_15A0680(a2, v8, 0);
        return sub_15A2C20(v16, v15, 1, 0);
      }
      v14 = 1;
      while ( v16 == sub_1580D00(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * v14), a2, 1) )
      {
        if ( ++v14 == v8 )
          goto LABEL_19;
      }
      v3 = *(_BYTE *)(a1 + 8);
    }
    if ( v3 != 15 || (unsigned __int8)sub_1642F90(*(_QWORD *)(a1 + 24), 1) )
      break;
    v11 = *(_DWORD *)(a1 + 8);
    v12 = sub_1644900(*(_QWORD *)a1, 1);
    v13 = sub_1646BA0(v12, v11 >> 8);
    a3 = 1;
    a1 = v13;
  }
  if ( !v7 )
    return 0;
  v9 = sub_15A4320(a1);
  v10 = sub_15FBEB0(v9, 0, a2, 0);
  return sub_15A46C0(v10, v9, a2, 0);
}
