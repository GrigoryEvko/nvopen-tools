// Function: sub_38F27B0
// Address: 0x38f27b0
//
__int64 __fastcall sub_38F27B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // rsi
  unsigned __int8 v7; // al
  __int64 v8; // rsi
  __int64 v9; // [rsp+0h] [rbp-90h] BYREF
  __int64 v10; // [rsp+8h] [rbp-88h]
  const char *v11; // [rsp+10h] [rbp-80h] BYREF
  char v12; // [rsp+20h] [rbp-70h]
  char v13; // [rsp+21h] [rbp-6Fh]
  const char *v14; // [rsp+30h] [rbp-60h] BYREF
  char v15; // [rsp+40h] [rbp-50h]
  char v16; // [rsp+41h] [rbp-4Fh]
  const char *v17; // [rsp+50h] [rbp-40h] BYREF
  char v18; // [rsp+60h] [rbp-30h]
  char v19; // [rsp+61h] [rbp-2Fh]

  if ( *(_BYTE *)(a1 + 845) || (result = sub_38E36C0(a1), !(_BYTE)result) )
  {
    v9 = 0;
    v10 = 0;
    v2 = sub_3909460(a1);
    v3 = sub_39092A0(v2);
    v6 = 0;
    if ( (unsigned __int8)sub_3909EB0(a1, 9) )
      goto LABEL_4;
    v13 = 1;
    v11 = "invalid option for '.bundle_lock' directive";
    v12 = 3;
    v7 = sub_38F0EE0(a1, &v9, v4, v5);
    if ( (unsigned __int8)sub_3909C80(a1, v7, v3, &v11) )
      return 1;
    v16 = 1;
    v8 = 1;
    v14 = "invalid option for '.bundle_lock' directive";
    v15 = 3;
    if ( v10 == 12 )
    {
      if ( *(_QWORD *)v9 != 0x6F745F6E67696C61LL || (v8 = 0, *(_DWORD *)(v9 + 8) != 1684956511) )
        v8 = 1;
    }
    if ( (unsigned __int8)sub_3909C80(a1, v8, v3, &v14) )
      return 1;
    v19 = 1;
    v18 = 3;
    v17 = "unexpected token after '.bundle_lock' directive option";
    v6 = 1;
    if ( (unsigned __int8)sub_3909E20(a1, 9, &v17) )
    {
      return 1;
    }
    else
    {
LABEL_4:
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 328) + 1032LL))(*(_QWORD *)(a1 + 328), v6);
      return 0;
    }
  }
  return result;
}
