// Function: sub_24646E0
// Address: 0x24646e0
//
unsigned __int64 __fastcall sub_24646E0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  _BYTE *v5; // r13
  __int64 *v6; // rdi
  __int64 **v7; // rax
  _BYTE *v9; // rax
  int v10; // [rsp+8h] [rbp-68h]
  _QWORD v11[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v12; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 8);
  v12 = 257;
  v5 = sub_94BCF0((unsigned int **)a2, *(_QWORD *)(v4 + 104), *(_QWORD *)(v4 + 80), (__int64)v11);
  if ( a3 )
  {
    v12 = 257;
    v9 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL), a3, 0);
    v5 = (_BYTE *)sub_929C50((unsigned int **)a2, v5, v9, (__int64)v11, 0, 0);
  }
  v6 = *(__int64 **)(a2 + 72);
  v11[0] = "_msarg";
  v12 = 259;
  v7 = (__int64 **)sub_BCE3C0(v6, 0);
  return sub_24633A0((__int64 *)a2, 0x30u, (unsigned __int64)v5, v7, (__int64)v11, 0, v10, 0);
}
