// Function: sub_946910
// Address: 0x946910
//
unsigned int *__fastcall sub_946910(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 i; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int *result; // rax
  unsigned __int64 v12; // [rsp+0h] [rbp-50h]
  unsigned int v13; // [rsp+8h] [rbp-48h]
  int v14; // [rsp+Ch] [rbp-44h]
  _DWORD v15[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = sub_BD5D20(a3);
  v6 = sub_C996C0("Generating Function IR", 22, v4, v5);
  v7 = *(_QWORD *)(a2 + 152);
  for ( i = v6; *(_BYTE *)(v7 + 140) == 12; v7 = *(_QWORD *)(v7 + 160) )
    ;
  v12 = *(_QWORD *)(v7 + 160);
  *(_QWORD *)(a1 + 528) = sub_91B910(a2, v15);
  v14 = dword_4D04658;
  v13 = dword_4D046B4;
  if ( !*(_DWORD *)(a2 + 64) || !*(_WORD *)(a2 + 68) )
  {
    dword_4D04658 = 1;
    dword_4D046B4 = 0;
  }
  sub_946060(a1, a2, v12, a3);
  if ( unk_4D046BC )
  {
    if ( unk_4D04660 )
    {
      sub_91D3A0(a1 + 248, *(_QWORD *)(*(_QWORD *)(a1 + 528) + 80LL));
      if ( !(unsigned __int8)sub_91CC30((unsigned __int8 *)(a1 + 248)) )
        *(_BYTE *)(a1 + 240) = 1;
    }
  }
  sub_9363D0((_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 528) + 80LL));
  sub_945FB0(a1, v7, v9, v10);
  dword_4D04658 = v14;
  result = &dword_4D046B4;
  dword_4D046B4 = v13;
  if ( i )
    return (unsigned int *)sub_C9AF60(i);
  return result;
}
