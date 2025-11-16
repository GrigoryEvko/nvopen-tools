// Function: sub_31F7FC0
// Address: 0x31f7fc0
//
__int64 __fastcall sub_31F7FC0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  int v9; // [rsp+Ch] [rbp-54h]
  __int16 v10; // [rsp+10h] [rbp-50h] BYREF
  int v11; // [rsp+12h] [rbp-4Eh]
  int v12; // [rsp+16h] [rbp-4Ah]
  __int64 v13; // [rsp+20h] [rbp-40h]
  __int64 v14; // [rsp+28h] [rbp-38h]
  __int64 v15; // [rsp+30h] [rbp-30h]

  v2 = *(_BYTE *)(a2 - 16);
  v3 = *(_QWORD *)(a2 + 24) >> 3;
  if ( (v2 & 2) != 0 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( v4 )
    {
LABEL_3:
      v4 = sub_B91420(v4);
      v6 = v5;
      goto LABEL_4;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a2 - 8LL * ((v2 >> 2) & 0xF));
    if ( v4 )
      goto LABEL_3;
  }
  v6 = 0;
LABEL_4:
  if ( sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1] >> 3 == 8 )
    v9 = 35;
  else
    v9 = 34;
  v13 = v3;
  v10 = 5379;
  v14 = v4;
  v11 = 112;
  v15 = v6;
  v12 = v9;
  v7 = sub_3709C80(a1 + 648, &v10);
  return sub_3707F80(a1 + 632, v7);
}
