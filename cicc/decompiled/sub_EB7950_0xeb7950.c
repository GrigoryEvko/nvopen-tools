// Function: sub_EB7950
// Address: 0xeb7950
//
__int64 __fastcall sub_EB7950(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rsi
  unsigned __int8 v5; // al
  __int64 v6; // rsi
  __int64 v7; // [rsp+0h] [rbp-90h] BYREF
  __int64 v8; // [rsp+8h] [rbp-88h]
  const char *v9; // [rsp+10h] [rbp-80h] BYREF
  char v10; // [rsp+30h] [rbp-60h]
  char v11; // [rsp+31h] [rbp-5Fh]
  const char *v12; // [rsp+40h] [rbp-50h] BYREF
  char v13; // [rsp+60h] [rbp-30h]
  char v14; // [rsp+61h] [rbp-2Fh]

  if ( *(_BYTE *)(a1 + 869) || (result = sub_EA2540(a1), !(_BYTE)result) )
  {
    v7 = 0;
    v8 = 0;
    v2 = sub_ECD7B0(a1);
    v3 = sub_ECD6A0(v2);
    v4 = 0;
    if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
      goto LABEL_4;
    v11 = 1;
    v9 = "invalid option for '.bundle_lock' directive";
    v10 = 3;
    v5 = sub_EB61F0(a1, &v7);
    if ( (unsigned __int8)sub_ECE070(a1, v5, v3, &v9) )
      return 1;
    v14 = 1;
    v6 = 1;
    v12 = "invalid option for '.bundle_lock' directive";
    v13 = 3;
    if ( v8 == 12 )
    {
      if ( *(_QWORD *)v7 != 0x6F745F6E67696C61LL || (v6 = 0, *(_DWORD *)(v7 + 8) != 1684956511) )
        v6 = 1;
    }
    if ( (unsigned __int8)sub_ECE070(a1, v6, v3, &v12) )
      return 1;
    v4 = 1;
    if ( (unsigned __int8)sub_ECE000(a1) )
    {
      return 1;
    }
    else
    {
LABEL_4:
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 232) + 1248LL))(*(_QWORD *)(a1 + 232), v4);
      return 0;
    }
  }
  return result;
}
