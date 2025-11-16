// Function: sub_1580EE0
// Address: 0x1580ee0
//
__int64 __fastcall sub_1580EE0(__int64 a1, __int64 a2, char a3)
{
  char v3; // al
  __int64 v4; // r12
  unsigned int v5; // eax
  char v7; // bl
  int v8; // r15d
  unsigned int v9; // r14d
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v3 = *(_BYTE *)(a1 + 8);
    if ( v3 == 14 )
    {
      a1 = *(_QWORD *)(a1 + 24);
LABEL_3:
      v4 = sub_15A4230(a1);
      v5 = sub_15FBEB0(v4, 0, a2, 0);
      return sub_15A46C0(v5, v4, a2, 0);
    }
    v7 = a3;
    if ( v3 == 13 )
    {
      if ( (*(_BYTE *)(a1 + 9) & 2) != 0 )
        return sub_15A0680(a2, 1, 0);
      v8 = *(_DWORD *)(a1 + 12);
      if ( !v8 )
        return sub_15A0680(a2, 1, 0);
      v13 = sub_1580EE0(**(_QWORD **)(a1 + 16), a2, 1);
      if ( v8 == 1 )
        return v13;
      v9 = 1;
      while ( v13 == sub_1580EE0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * v9), a2, 1) )
      {
        if ( v8 == ++v9 )
          return v13;
      }
      v3 = *(_BYTE *)(a1 + 8);
    }
    if ( v3 != 15 || (unsigned __int8)sub_1642F90(*(_QWORD *)(a1 + 24), 1) )
      break;
    v10 = *(_DWORD *)(a1 + 8);
    v11 = sub_1644900(*(_QWORD *)a1, 1);
    v12 = sub_1646BA0(v11, v10 >> 8);
    a3 = 1;
    a1 = v12;
  }
  if ( v7 )
    goto LABEL_3;
  return 0;
}
