// Function: sub_693280
// Address: 0x693280
//
__int64 __fastcall sub_693280(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r13d
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 result; // rax
  unsigned int v17; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = 0;
  v18[0] = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v9 = v18[0];
  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
  {
    v7 = 1;
    sub_7296C0(&v17);
    v9 = v18[0];
  }
  sub_72BAF0(v9, a3, unk_4F06A51);
  v10 = sub_73A460(v18[0]);
  v11 = sub_73A720(v10);
  sub_724E30(v18);
  v12 = sub_731250(a1);
  if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(a1 + 120)) )
  {
    *(_BYTE *)(v12 + 25) &= ~1u;
    v12 = sub_73DDB0(v12);
  }
  v13 = sub_6EE5A0(v12);
  *(_QWORD *)(v13 + 16) = v11;
  v14 = sub_73DBF0(92, a2[15], v13);
  *(_BYTE *)(v14 + 25) |= 1u;
  if ( unk_4D04810 )
    *(_BYTE *)(v14 + 60) |= 1u;
  *((_BYTE *)a2 + 177) = 5;
  v15 = *a2;
  a2[23] = v14;
  result = sub_8756B0(v15);
  if ( v7 )
    return sub_729730(v17);
  return result;
}
