// Function: sub_2AC7650
// Address: 0x2ac7650
//
__int64 __fastcall sub_2AC7650(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // r10d
  __int64 v4; // r11
  __int16 v5; // r15
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // rax
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  int v17; // [rsp+18h] [rbp-68h]
  unsigned __int8 v18; // [rsp+1Fh] [rbp-61h]
  const char *v19[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v20; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 904);
  v3 = *(_DWORD *)(v2 + 104);
  v4 = *(_QWORD *)(v2 + 96);
  v5 = *(_WORD *)(v2 + 108);
  v18 = *(_BYTE *)(v2 + 110);
  v6 = *(_QWORD *)(a1 + 160);
  if ( v6 )
  {
    v7 = *(_BYTE *)(v6 + 1) >> 1;
    if ( v7 == 127 )
      v7 = -1;
    *(_DWORD *)(v2 + 104) = v7;
  }
  v8 = *(_QWORD *)(a1 + 48);
  BYTE4(v19[0]) = 0;
  LODWORD(v19[0]) = 0;
  v17 = v3;
  v14 = v4;
  v9 = sub_2BFB120(a2, *(_QWORD *)(v8 + 16), v19);
  BYTE4(v19[0]) = 0;
  v15 = v9;
  v10 = *(_QWORD *)(a1 + 48);
  LODWORD(v19[0]) = 0;
  v11 = sub_2BFB120(a2, *(_QWORD *)(v10 + 8), v19);
  v12 = sub_2ABE630(
          *(_QWORD *)(a2 + 904),
          v11,
          *(_QWORD *)(**(_QWORD **)(a1 + 48) + 40LL),
          v15,
          *(_DWORD *)(a1 + 152),
          *(unsigned __int8 **)(a1 + 160));
  v20 = 260;
  v19[0] = (const char *)(a1 + 168);
  v16 = (__int64)v12;
  sub_BD6B50(v12, v19);
  LODWORD(v19[0]) = 0;
  BYTE4(v19[0]) = 0;
  sub_2AC6E90(a2, a1 + 96, v16, (unsigned int *)v19);
  *(_WORD *)(v2 + 108) = v5;
  *(_QWORD *)(v2 + 96) = v14;
  *(_DWORD *)(v2 + 104) = v17;
  *(_BYTE *)(v2 + 110) = v18;
  return v18;
}
