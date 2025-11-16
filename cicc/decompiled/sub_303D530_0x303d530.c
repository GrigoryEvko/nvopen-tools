// Function: sub_303D530
// Address: 0x303d530
//
__int64 __fastcall sub_303D530(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int16 v7; // bx
  char v8; // al
  __int64 result; // rax
  __int64 v10; // r15
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+8h] [rbp-78h]
  __int128 v16; // [rsp+10h] [rbp-70h] BYREF
  __int128 v17; // [rsp+20h] [rbp-60h]
  __int64 v18; // [rsp+30h] [rbp-50h] BYREF
  int v19; // [rsp+38h] [rbp-48h]
  __int64 v20; // [rsp+40h] [rbp-40h]
  int v21; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v7 = *(_WORD *)v6;
  if ( *(_WORD *)v6 == 2 )
    return sub_303D400(a1, a2, a3, a4);
  v8 = sub_307AB50(v7, *(_QWORD *)(v6 + 8));
  if ( v7 != 37 && v8 != 1 )
    return 0;
  v10 = *(_QWORD *)(a2 + 104);
  v11 = *(unsigned __int16 *)(a2 + 96);
  v14 = *(_QWORD *)(a2 + 112);
  v12 = sub_2E79000(*(__int64 **)(a4 + 40));
  if ( (unsigned __int8)sub_2FEBAB0(a1, *(_QWORD *)(a4 + 64), v12, v11, v10, v14, 0) )
    return 0;
  v16 = 0;
  v17 = 0;
  sub_3461D10(&v18, a1, a2, a4);
  v13 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v16 = v18;
  v18 = v13;
  DWORD2(v16) = v19;
  *(_QWORD *)&v17 = v20;
  DWORD2(v17) = v21;
  if ( v13 )
    sub_B96E90((__int64)&v18, v13, 1);
  v19 = *(_DWORD *)(a2 + 72);
  result = sub_3411660(a4, &v16, 2, &v18);
  if ( v18 )
  {
    v15 = result;
    sub_B91220((__int64)&v18, v18);
    return v15;
  }
  return result;
}
