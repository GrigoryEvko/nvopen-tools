// Function: sub_EB5C40
// Address: 0xeb5c40
//
__int64 __fastcall sub_EB5C40(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rbx
  const char *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rbx
  _QWORD v8[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 328);
  if ( *(_QWORD *)(a1 + 320) != v2 && *(_BYTE *)(v2 - 3) )
  {
    sub_EB4E00(a1);
    return 0;
  }
  if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
  {
    v4 = 41;
    v5 = ".warning directive invoked in source file";
LABEL_7:
    v8[0] = v5;
    v8[1] = v4;
    v9 = 261;
    return sub_EA8060((_QWORD *)a1, a2, (__int64)v8, 0, 0);
  }
  if ( **(_DWORD **)(a1 + 48) != 3 )
  {
    v8[0] = ".warning argument must be a string";
    v9 = 259;
    return sub_ECE0E0(a1, v8, 0, 0);
  }
  v6 = sub_ECD7B0(a1);
  v4 = *(_QWORD *)(v6 + 16);
  v5 = *(const char **)(v6 + 8);
  if ( v4 )
  {
    v7 = v4 - 1;
    if ( !v7 )
      v7 = 1;
    ++v5;
    v4 = v7 - 1;
  }
  sub_EABFE0(a1);
  result = sub_ECE000(a1);
  if ( !(_BYTE)result )
    goto LABEL_7;
  return result;
}
