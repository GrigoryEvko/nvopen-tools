// Function: sub_3268160
// Address: 0x3268160
//
__int64 __fastcall sub_3268160(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // r15
  unsigned __int16 *v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // r10
  int v11; // r8d
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int128 v16; // [rsp-20h] [rbp-80h]
  __int128 v17; // [rsp-10h] [rbp-70h]
  int v18; // [rsp+0h] [rbp-60h]
  int v19; // [rsp+8h] [rbp-58h]
  int v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 40);
  v4 = *v3;
  v5 = v3[1];
  v6 = v3[5];
  v7 = v3[6];
  if ( !(unsigned __int8)sub_33CF170(v3[10], v3[11]) )
    return 0;
  v8 = *(unsigned __int16 **)(a2 + 48);
  if ( *((_BYTE *)a1 + 33) )
  {
    v13 = *v8;
    v14 = a1[1];
    v15 = 1;
    if ( (_WORD)v13 != 1 )
    {
      if ( !(_WORD)v13 )
        return 0;
      v15 = (unsigned __int16)v13;
      if ( !*(_QWORD *)(v14 + 8 * v13 + 112) )
        return 0;
    }
    if ( (*(_BYTE *)(v14 + 500 * v15 + 6492) & 0xFB) != 0 )
      return 0;
  }
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *a1;
  v11 = *(_DWORD *)(a2 + 68);
  v23 = v9;
  if ( v9 )
  {
    v18 = v11;
    v19 = (int)v8;
    v20 = v10;
    sub_B96E90((__int64)&v23, v9, 1);
    v11 = v18;
    LODWORD(v8) = v19;
    LODWORD(v10) = v20;
  }
  *((_QWORD *)&v17 + 1) = v7;
  *(_QWORD *)&v17 = v6;
  *((_QWORD *)&v16 + 1) = v5;
  *(_QWORD *)&v16 = v4;
  v24 = *(_DWORD *)(a2 + 72);
  result = sub_3411F20(v10, 78, (unsigned int)&v23, (_DWORD)v8, v11, (unsigned int)&v23, v16, v17);
  if ( v23 )
  {
    v21 = result;
    sub_B91220((__int64)&v23, v23);
    return v21;
  }
  return result;
}
