// Function: sub_30B3900
// Address: 0x30b3900
//
__int64 *__fastcall sub_30B3900(__int64 *a1, __int64 a2, __int64 a3)
{
  int v5; // r15d
  _QWORD *v6; // rdi
  __int64 v7; // rax
  _WORD *v8; // rdx
  unsigned __int64 *v9; // rax
  unsigned __int64 v11[2]; // [rsp+0h] [rbp-90h] BYREF
  _BYTE v12[16]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v13[3]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v14; // [rsp+38h] [rbp-58h]
  _QWORD *v15; // [rsp+40h] [rbp-50h]
  __int64 v16; // [rsp+48h] [rbp-48h]
  unsigned __int64 *v17; // [rsp+50h] [rbp-40h]

  v16 = 0x100000000LL;
  v17 = v11;
  v11[0] = (unsigned __int64)v12;
  v13[0] = &unk_49DD210;
  v11[1] = 0;
  v12[0] = 0;
  v13[1] = 0;
  v13[2] = 0;
  v14 = 0;
  v15 = 0;
  sub_CB5980((__int64)v13, 0, 0, 0);
  v5 = *(_DWORD *)(a3 + 8);
  if ( (unsigned __int64)(v14 - (_QWORD)v15) <= 7 )
  {
    v6 = (_QWORD *)sub_CB6200((__int64)v13, "label=\"[", 8u);
  }
  else
  {
    v6 = v13;
    *v15++ = 0x5B223D6C6562616CLL;
  }
  v7 = sub_30B0E00((__int64)v6, v5);
  v8 = *(_WORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v7, (unsigned __int8 *)"]\"", 2u);
  }
  else
  {
    *v8 = 8797;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v9 = v17;
  *a1 = (__int64)(a1 + 2);
  sub_30B3180(a1, (_BYTE *)*v9, *v9 + v9[1]);
  v13[0] = &unk_49DD210;
  sub_CB5840((__int64)v13);
  if ( (_BYTE *)v11[0] != v12 )
    j_j___libc_free_0(v11[0]);
  return a1;
}
