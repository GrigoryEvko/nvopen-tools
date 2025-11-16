// Function: sub_10E5130
// Address: 0x10e5130
//
unsigned __int8 *__fastcall sub_10E5130(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *result; // rax
  __int64 v4; // r11
  __int64 v5; // r10
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // [rsp-40h] [rbp-F0h]
  __int64 v10; // [rsp-38h] [rbp-E8h]
  __int64 v11; // [rsp-30h] [rbp-E0h]
  __int128 v12; // [rsp-28h] [rbp-D8h]
  __int128 v13; // [rsp-18h] [rbp-C8h]
  _QWORD *v14; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD *v15; // [rsp+8h] [rbp-A8h] BYREF
  _BYTE v16[160]; // [rsp+10h] [rbp-A0h] BYREF

  if ( !(unsigned __int8)sub_F0C3C0((__int64)a1) )
    return 0;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 )
    return 0;
  if ( *(_BYTE *)v2 )
    return 0;
  if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v2 + 24) )
    return 0;
  if ( (unsigned __int16)((*(_WORD *)(a2 + 2) & 3) - 2) <= 1u )
    return 0;
  v4 = a1[9];
  v5 = a1[11];
  v6 = a1[10];
  *((_QWORD *)&v13 + 1) = &v15;
  v7 = a1[8];
  *(_QWORD *)&v13 = sub_10E9120;
  *((_QWORD *)&v12 + 1) = &v14;
  *(_QWORD *)&v12 = sub_10E5250;
  v11 = a1[24];
  v10 = a1[22];
  v9 = a1[21];
  v14 = a1;
  v15 = a1;
  sub_11EE200((unsigned int)v16, v5, v4, v6, (_DWORD)a1 + 200, v7, v9, v10, v11, v12, v13);
  v8 = sub_11F2320(v16, a2, a1[4]);
  if ( !v8 )
    return 0;
  result = (unsigned __int8 *)a2;
  if ( *(_QWORD *)(a2 + 16) )
    return sub_F162A0((__int64)a1, a2, v8);
  return result;
}
