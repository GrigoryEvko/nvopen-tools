// Function: sub_1281C00
// Address: 0x1281c00
//
__int64 __fastcall sub_1281C00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r12
  unsigned __int8 v7; // al
  unsigned int v9; // r14d
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 *v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+18h] [rbp-48h] BYREF
  char v20[16]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v21; // [rsp+30h] [rbp-30h]

  v5 = (_QWORD *)a2;
  v7 = *(_BYTE *)(a3 + 16);
  if ( v7 > 0x10u )
    goto LABEL_8;
  if ( v7 == 13 )
  {
    v9 = *(_DWORD *)(a3 + 32);
    if ( v9 <= 0x40 )
    {
      if ( *(_QWORD *)(a3 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) )
        goto LABEL_3;
    }
    else
    {
      v18 = a3;
      v10 = sub_16A58F0(a3 + 24);
      a3 = v18;
      if ( v9 != v10 )
        goto LABEL_3;
    }
    return (__int64)v5;
  }
LABEL_3:
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A2CF0(a2, a3);
LABEL_8:
  v21 = 257;
  v11 = sub_15FB440(26, a2, a3, v20, 0);
  v12 = a1[1];
  v5 = (_QWORD *)v11;
  if ( v12 )
  {
    v13 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v12 + 40, v11);
    v14 = v5[3];
    v15 = *v13;
    v5[4] = v13;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v15 | v14 & 7;
    *(_QWORD *)(v15 + 8) = v5 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v5 + 3);
  }
  sub_164B780(v5, a4);
  v16 = *a1;
  if ( !*a1 )
    return (__int64)v5;
  v19 = *a1;
  sub_1623A60(&v19, v16, 2);
  if ( v5[6] )
    sub_161E7C0(v5 + 6);
  v17 = v19;
  v5[6] = v19;
  if ( !v17 )
    return (__int64)v5;
  sub_1623210(&v19, v17, v5 + 6);
  return (__int64)v5;
}
