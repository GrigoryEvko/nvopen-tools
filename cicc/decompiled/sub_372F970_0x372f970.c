// Function: sub_372F970
// Address: 0x372f970
//
__int64 __fastcall sub_372F970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int16 v6; // ax
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // [rsp+8h] [rbp-68h] BYREF
  __int64 v10; // [rsp+10h] [rbp-60h] BYREF
  char v11; // [rsp+40h] [rbp-30h]

  result = sub_372F900(*(_QWORD *)a1, *(_QWORD *)(a1 + 8), a3, a4, a5);
  if ( (_BYTE)result )
  {
    v6 = *(_WORD *)(a1 + 32);
    v7 = *(_QWORD *)(a1 + 16);
    v8 = *(_QWORD *)(a1 + 24);
    v11 = 2;
    WORD2(v10) = v6;
    v9 = v7 + 40;
    LODWORD(v10) = v8;
    sub_372F890(&v9, &v10);
    if ( v11 != -1 )
      funcs_32198D3[v11]();
    result = *(unsigned __int8 *)(v7 + 88);
    if ( (_BYTE)result != 2 )
      sub_435074((_BYTE)result == 0xFF);
  }
  return result;
}
