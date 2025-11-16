// Function: sub_12A5570
// Address: 0x12a5570
//
__int64 __fastcall sub_12A5570(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 i; // r14
  int v13; // r15d
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 result; // rax
  unsigned __int64 v18; // [rsp+0h] [rbp-50h]
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  _DWORD v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = sub_1649960(a3);
  v8 = v7;
  if ( sub_16DA870(a3, a2, v7, v9, v10, v11) )
    sub_16DB3F0("Generating Function IR", 22, v6, v8);
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v18 = *(_QWORD *)(i + 160);
  *(_QWORD *)(a1 + 440) = sub_127B5C0(a2, v20);
  v13 = dword_4D04658;
  v19 = dword_4D046B4;
  if ( !*(_DWORD *)(a2 + 64) || !*(_WORD *)(a2 + 68) )
  {
    dword_4D04658 = 1;
    dword_4D046B4 = 0;
  }
  sub_12A5110(a1, a2, v18, a3);
  if ( unk_4D046BC )
  {
    if ( unk_4D04660 )
    {
      sub_127CDE0(a1 + 176, *(_QWORD *)(*(_QWORD *)(a1 + 440) + 80LL));
      if ( !(unsigned __int8)sub_127C8E0((unsigned __int8 *)(a1 + 176)) )
        *(_BYTE *)(a1 + 168) = 1;
    }
  }
  sub_1296350((__int64 *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 440) + 80LL));
  sub_12A5080((_QWORD *)a1, i);
  dword_4D04658 = v13;
  dword_4D046B4 = v19;
  result = sub_16DA870(a1, i, v14, v19, v15, v16);
  if ( result )
    return sub_16DB5E0();
  return result;
}
