// Function: sub_30B4090
// Address: 0x30b4090
//
__int64 *__fastcall sub_30B4090(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // r8d
  unsigned __int64 *v6; // rax
  int v8; // [rsp+4h] [rbp-BCh]
  unsigned __int64 v10[2]; // [rsp+10h] [rbp-B0h] BYREF
  _BYTE v11[16]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int8 *v12[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v13; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v14[3]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v15; // [rsp+68h] [rbp-58h]
  _QWORD *v16; // [rsp+70h] [rbp-50h]
  __int64 v17; // [rsp+78h] [rbp-48h]
  unsigned __int64 *v18; // [rsp+80h] [rbp-40h]

  v17 = 0x100000000LL;
  v18 = v10;
  v10[0] = (unsigned __int64)v11;
  v14[0] = &unk_49DD210;
  v10[1] = 0;
  v11[0] = 0;
  v14[1] = 0;
  v14[2] = 0;
  v15 = 0;
  v16 = 0;
  sub_CB5980((__int64)v14, 0, 0, 0);
  v5 = *(_DWORD *)(a3 + 8);
  if ( (unsigned __int64)(v15 - (_QWORD)v16) <= 7 )
  {
    v8 = *(_DWORD *)(a3 + 8);
    sub_CB6200((__int64)v14, "label=\"[", 8u);
    v5 = v8;
  }
  else
  {
    *v16++ = 0x5B223D6C6562616CLL;
  }
  if ( v5 == 2 )
  {
    sub_30B3E50((__int64)v12, a4, a2, *(_QWORD *)a3);
    sub_CB6200((__int64)v14, v12[0], (size_t)v12[1]);
    if ( (__int64 *)v12[0] != &v13 )
      j_j___libc_free_0((unsigned __int64)v12[0]);
  }
  else
  {
    sub_30B0E00((__int64)v14, v5);
  }
  if ( (unsigned __int64)(v15 - (_QWORD)v16) <= 1 )
  {
    sub_CB6200((__int64)v14, (unsigned __int8 *)"]\"", 2u);
  }
  else
  {
    *(_WORD *)v16 = 8797;
    v16 = (_QWORD *)((char *)v16 + 2);
  }
  v6 = v18;
  *a1 = (__int64)(a1 + 2);
  sub_30B3180(a1, (_BYTE *)*v6, *v6 + v6[1]);
  v14[0] = &unk_49DD210;
  sub_CB5840((__int64)v14);
  if ( (_BYTE *)v10[0] != v11 )
    j_j___libc_free_0(v10[0]);
  return a1;
}
