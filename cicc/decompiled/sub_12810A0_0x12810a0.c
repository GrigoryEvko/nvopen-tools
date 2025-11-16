// Function: sub_12810A0
// Address: 0x12810a0
//
_QWORD *__fastcall sub_12810A0(__int64 *a1, unsigned __int64 a2, unsigned int a3, char a4)
{
  unsigned int v4; // r15d
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 *v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  char *v16; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+21h] [rbp-3Fh]

  v4 = 1;
  v18 = 1;
  v16 = "tmp";
  v17 = 3;
  if ( !a4 )
  {
    v4 = unk_4D0463C;
    if ( unk_4D0463C )
      v4 = sub_126A420(a1[4], a2);
  }
  v7 = (_QWORD *)sub_1648A60(64, 1);
  if ( v7 )
    sub_15F9210(v7, *(_QWORD *)(*(_QWORD *)a2 + 24LL), a2, 0, v4, 0);
  v8 = a1[7];
  if ( v8 )
  {
    v9 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v8 + 40, v7);
    v10 = v7[3];
    v11 = *v9;
    v7[4] = v9;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    v7[3] = v11 | v10 & 7;
    *(_QWORD *)(v11 + 8) = v7 + 3;
    *v9 = *v9 & 7 | (unsigned __int64)(v7 + 3);
  }
  sub_164B780(v7, &v16);
  v12 = a1[6];
  if ( v12 )
  {
    v15 = a1[6];
    sub_1623A60(&v15, v12, 2);
    if ( v7[6] )
      sub_161E7C0(v7 + 6);
    v13 = v15;
    v7[6] = v15;
    if ( v13 )
      sub_1623210(&v15, v13, v7 + 6);
  }
  sub_15F8F50(v7, a3);
  return v7;
}
