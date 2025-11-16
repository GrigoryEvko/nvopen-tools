// Function: sub_6320D0
// Address: 0x6320d0
//
__int64 __fastcall sub_6320D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 result; // rax
  int *v8; // rdx
  __int64 v9; // r15
  char v10; // al
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1;
  if ( *(_BYTE *)(a1 + 8) == 1 )
  {
    v11 = *(_QWORD **)(a1 + 24);
    if ( v11 )
    {
      if ( !*v11 )
        v5 = *(_QWORD *)(a1 + 24);
    }
  }
  result = sub_6E1AD0(v5, v17);
  if ( (_DWORD)result )
  {
    v8 = &v16;
    v16 = 0;
    v9 = *(_QWORD *)(v17[0] + 128);
    if ( !dword_4F077C0 )
      v8 = 0;
    if ( !(unsigned int)sub_631DE0(a2, v17[0], v8) )
    {
      v10 = *(_BYTE *)(a3 + 40);
      if ( (v10 & 0x20) != 0 )
      {
        *(_BYTE *)(a3 + 41) |= 2u;
      }
      else
      {
        v15 = *a2;
        v14 = sub_6E1A20(v5);
        sub_6861A0(144, v14, v9, v15);
        v10 = *(_BYTE *)(a3 + 40);
      }
      if ( (v10 & 0x40) == 0 )
        *(_QWORD *)a4 = sub_72C9A0();
      if ( (unsigned int)sub_8D23E0(*a2) )
      {
        if ( (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
          *a2 = sub_72C930();
      }
      return 1;
    }
    if ( (*(_BYTE *)(a3 + 40) & 0x40) == 0 )
    {
      *(_QWORD *)a4 = sub_740630(v17[0]);
      *(_QWORD *)(*(_QWORD *)a4 + 64LL) = *(_QWORD *)sub_6E1A20(v5);
      if ( *(_BYTE *)(v5 + 8) != 2 )
        *(_QWORD *)(*(_QWORD *)a4 + 112LL) = *(_QWORD *)sub_6E1A60(v5);
    }
    *(_BYTE *)(a3 + 41) = (*(_BYTE *)(v17[0] + 170) >> 2) & 0x10 | *(_BYTE *)(a3 + 41) & 0xEF;
    if ( dword_4D04964 && !dword_4D04428 )
    {
      if ( (*(_BYTE *)(a3 + 40) & 0x20) != 0 )
        return 1;
      if ( (unsigned int)sub_6E1B20(v5) )
      {
        v12 = sub_6E1A20(v5);
        sub_684AA0(unk_4F07471, 1584, v12);
        return 1;
      }
    }
    if ( v16 && (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
    {
      v13 = sub_6E1A20(v5);
      sub_684B30(1828, v13);
    }
    return 1;
  }
  return result;
}
