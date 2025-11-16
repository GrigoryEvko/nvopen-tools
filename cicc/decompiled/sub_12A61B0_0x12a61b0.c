// Function: sub_12A61B0
// Address: 0x12a61b0
//
__int64 __fastcall sub_12A61B0(__int64 *a1, __int64 a2, unsigned __int64 a3, unsigned int a4, char a5)
{
  int v6; // ecx
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-64h]
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  char v20[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v6 = 1;
  if ( !a5 )
  {
    v6 = unk_4D0463C;
    if ( unk_4D0463C )
      v6 = sub_126A420(a1[4], a3);
  }
  v18 = v6;
  v21 = 257;
  v9 = sub_1648A60(64, 2);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15F9650(v9, a2, a3, v18, 0);
  v11 = a1[7];
  if ( v11 )
  {
    v12 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v11 + 40, v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, v20);
  v15 = a1[6];
  if ( v15 )
  {
    v19 = a1[6];
    sub_1623A60(&v19, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v19;
    v10[6] = v19;
    if ( v16 )
      sub_1623210(&v19, v16, v10 + 6);
  }
  return sub_15F9450(v10, a4);
}
