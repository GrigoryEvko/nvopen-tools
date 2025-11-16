// Function: sub_6E3BA0
// Address: 0x6e3ba0
//
__int64 __fastcall sub_6E3BA0(__int64 a1, __int64 *a2, int a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // rax
  _QWORD *v7; // [rsp-28h] [rbp-28h]
  _QWORD *v8; // [rsp-28h] [rbp-28h]
  int v9; // [rsp-1Ch] [rbp-1Ch]
  int v10; // [rsp-1Ch] [rbp-1Ch]

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) == 0 )
    return result;
  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 1 )
  {
    v5 = *(_QWORD *)(a1 + 144);
    goto LABEL_6;
  }
  if ( (_BYTE)result != 2 )
    return result;
  v5 = *(_QWORD *)(a1 + 288);
  if ( v5 )
  {
LABEL_7:
    v7 = a4;
    v9 = a3;
    v6 = sub_6E36E0(v5);
    return sub_6E3AC0(v6, a2, v9, v7);
  }
  if ( *(_BYTE *)(a1 + 317) == 12 && *(_BYTE *)(a1 + 320) == 1 )
  {
    v8 = a4;
    v10 = a3;
    result = sub_72E9A0(a1 + 144);
    a4 = v8;
    a3 = v10;
    v5 = result;
LABEL_6:
    if ( v5 )
      goto LABEL_7;
  }
  return result;
}
