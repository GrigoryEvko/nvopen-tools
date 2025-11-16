// Function: sub_131BF20
// Address: 0x131bf20
//
__int64 __fastcall sub_131BF20(
        _BYTE *a1,
        unsigned int a2,
        __int64 (__fastcall **a3)(int, int, int, int, int, int, int),
        char a4)
{
  __int64 (__fastcall **v4)(int, int, int, int, int, int, int); // r14
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // r14
  __int64 v10; // rdi
  _BOOL8 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // [rsp+8h] [rbp-78h]
  int v16; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h] BYREF
  __int64 (__fastcall **v19[8])(int, int, int, int, int, int, int); // [rsp+40h] [rbp-40h] BYREF

  v4 = a3;
  v16 = 0;
  v17 = 0;
  if ( !a4 )
    v4 = &off_49E8020;
  sub_1340E90(v19, v4, a2);
  v5 = sub_131B220(a1, 0, v19, &v16, &v17, 3912, 16);
  v6 = v5;
  if ( !v5 )
    return 0;
  v14 = (__int64)(v5 + 2);
  v7 = sub_131B1D0(v5 + 2, &v18, 3968, 64);
  sub_1340E90(v7, a3, a2);
  sub_1340E90(v7 + 16, v4, a2);
  if ( (unsigned __int8)sub_130AF40(v7 + 32) )
  {
    v7 = 0;
    sub_131B7C0(a1, (unsigned int *)v19, v6, *v6);
  }
  else
  {
    v8 = v16;
    *(_QWORD *)(v7 + 160) = v6;
    v9 = v7 + 168;
    *(_BYTE *)(v7 + 144) = 0;
    *(_DWORD *)(v7 + 148) = v8;
    *(_QWORD *)(v7 + 152) = v17;
    do
    {
      v10 = v9;
      v9 += 16;
      sub_133F510(v10, "base");
    }
    while ( v9 != v7 + 3880 );
    *(_QWORD *)(v7 + 3880) = 144;
    *(_QWORD *)(v7 + 3888) = 4096;
    *(_QWORD *)(v7 + 3896) = *v6;
    v11 = 0;
    if ( dword_4F96B94 == 2 )
      v11 = unk_505F9C8 == 0;
    v12 = v18;
    *(_QWORD *)(v7 + 3904) = v11;
    sub_131B080(v7, v14, v12, v7, 3968);
  }
  return v7;
}
