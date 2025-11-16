// Function: sub_2F41240
// Address: 0x2f41240
//
__int64 __fastcall sub_2F41240(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r13d
  int v7; // edx
  int v8; // eax
  unsigned int v9; // eax
  int v10; // eax
  unsigned int v11; // r12d
  unsigned int v13; // edx
  unsigned int v14; // ebx
  __int64 v15; // rdi

  v6 = *(unsigned __int16 *)(a4 + 12);
  if ( *(_BYTE *)(a4 + 16) && (*(_BYTE *)(a3 + 3) & 0x10) == 0 )
    *(_BYTE *)(a3 + 4) |= 1u;
  v7 = (*(_DWORD *)a3 >> 8) & 0xFFF;
  if ( ((*(_DWORD *)a3 >> 8) & 0xFFF) == 0 )
  {
    sub_2EAB0C0(a3, v6);
    v15 = a3;
    v11 = 0;
    sub_2EAB350(v15, *(_BYTE *)(a4 + 16) ^ 1);
    return v11;
  }
  v8 = sub_E91CF0(*(_QWORD **)(a1 + 16), v6, v7);
  sub_2EAB0C0(a3, v8);
  sub_2EAB350(a3, *(_BYTE *)(a4 + 16) ^ 1);
  v9 = *(unsigned __int8 *)(a3 + 3);
  if ( (v9 & 0x10) == 0 )
  {
    v10 = *(_DWORD *)a3 >> 30;
    *(_DWORD *)a3 &= 0xFFF000FF;
    v11 = v10 & 1;
    if ( (v10 & 1) != 0 )
    {
      sub_2E8F280(a2, v6, *(_QWORD **)(a1 + 16), 1);
      return v11;
    }
    return 0;
  }
  v11 = *(_BYTE *)(a3 + 4) & 1;
  if ( !v11 )
    return 0;
  v13 = v9;
  LOBYTE(v13) = (unsigned __int8)v9 >> 4;
  v14 = v13;
  LOBYTE(v14) = ((v9 & 0x40) != 0) & ((unsigned __int8)v9 >> 4);
  if ( (_BYTE)v14 )
  {
    sub_2E8F690(a2, v6, *(_QWORD **)(a1 + 16), 1);
    return v14;
  }
  else
  {
    sub_2E8FA40(a2, v6, *(_QWORD *)(a1 + 16));
  }
  return v11;
}
