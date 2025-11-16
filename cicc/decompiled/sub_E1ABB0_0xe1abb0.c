// Function: sub_E1ABB0
// Address: 0xe1abb0
//
__int64 __fastcall sub_E1ABB0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rdx
  __int64 v11; // r15
  unsigned __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 result; // rax
  char v15; // dl
  __int64 v16; // [rsp+8h] [rbp-38h]

  v6 = sub_E18BB0(a1);
  if ( !v6 )
    return 0;
  v10 = *(_QWORD **)(a1 + 4912);
  v11 = v6;
  v12 = v10[1] + 48LL;
  if ( v12 > 0xFEF )
  {
    v16 = *(_QWORD *)(a1 + 4912);
    v13 = (_QWORD *)malloc(4096, a2, v10, v7, v8, v9);
    if ( !v13 )
      sub_2207530(4096, a2, v16);
    *v13 = v16;
    v10 = v13;
    v13[1] = 0;
    *(_QWORD *)(a1 + 4912) = v13;
    v12 = 48;
  }
  v10[1] = v12;
  result = *(_QWORD *)(a1 + 4912) + *(_QWORD *)(*(_QWORD *)(a1 + 4912) + 8LL) - 32LL;
  v15 = *(_BYTE *)(result + 10);
  *(_WORD *)(result + 8) = ((a4 & 0x3F) << 8) | 0x4042;
  *(_QWORD *)(result + 16) = a2;
  *(_QWORD *)(result + 24) = a3;
  *(_QWORD *)(result + 32) = v11;
  *(_BYTE *)(result + 10) = v15 & 0xF0 | 5;
  *(_QWORD *)result = &unk_49E0688;
  return result;
}
