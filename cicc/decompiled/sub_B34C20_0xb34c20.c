// Function: sub_B34C20
// Address: 0xb34c20
//
__int64 __fastcall sub_B34C20(__int64 a1, __int64 **a2, __int64 a3, char a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v16; // rax
  char v17; // [rsp+7h] [rbp-69h]
  __int64 v18; // [rsp+8h] [rbp-68h]
  _QWORD v19[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v20[10]; // [rsp+20h] [rbp-50h] BYREF

  v10 = a6;
  v11 = *(_QWORD *)(a3 + 8);
  if ( !a6 )
  {
    v17 = a4;
    v18 = v11;
    v16 = sub_ACADE0(a2);
    a4 = v17;
    v11 = v18;
    v10 = v16;
  }
  v20[0] = a3;
  v12 = *(_QWORD *)(a1 + 72);
  v19[0] = a2;
  v19[1] = v11;
  v13 = (unsigned int)(1LL << a4);
  v14 = sub_BCB2D0(v12);
  v20[2] = a5;
  v20[3] = v10;
  v20[1] = sub_ACD640(v14, v13, 0);
  return sub_B34BE0(a1, 0xE4u, (int)v20, 4, (__int64)v19, 2, a7);
}
