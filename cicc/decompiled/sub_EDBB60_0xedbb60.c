// Function: sub_EDBB60
// Address: 0xedbb60
//
__int64 *__fastcall sub_EDBB60(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  __int64 v10; // [rsp+18h] [rbp-58h] BYREF
  void *v11[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v12; // [rsp+40h] [rbp-30h]

  v3 = *a3;
  v8 = a2;
  *a3 = 0;
  v9 = 0;
  v10 = v3 | 1;
  sub_EDB980((__int64 *)v11, &v10, (__int64)&v8);
  if ( ((unsigned __int64)v11[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  if ( (v10 & 1) != 0 || (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v10, (__int64)&v10);
  if ( (v9 & 1) != 0 || (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v9, (__int64)&v10);
  v4 = *(_DWORD *)(a2 + 8);
  v11[0] = (void *)(a2 + 16);
  v12 = 260;
  v5 = sub_22077B0(48);
  v6 = v5;
  if ( v5 )
  {
    *(_DWORD *)(v5 + 8) = v4;
    *(_QWORD *)v5 = &unk_49E4BC8;
    sub_CA0F50((__int64 *)(v5 + 16), v11);
  }
  *a1 = v6 | 1;
  return a1;
}
