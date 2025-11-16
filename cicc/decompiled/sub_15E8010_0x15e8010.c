// Function: sub_15E8010
// Address: 0x15e8010
//
_QWORD *__fastcall sub_15E8010(__int64 *a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+10h] [rbp-70h]
  __int64 **v18; // [rsp+18h] [rbp-68h]
  __int64 v19[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v20[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = a5;
  v11 = *a2;
  v12 = *(_QWORD *)(*a2 + 24);
  if ( !a5 )
  {
    v17 = *a2;
    v18 = *(__int64 ***)(*a2 + 24);
    v16 = sub_1599EF0(v18);
    v11 = v17;
    v12 = (__int64)v18;
    v9 = v16;
  }
  v13 = a1[3];
  v19[0] = v12;
  v19[1] = v11;
  v20[0] = a2;
  v14 = sub_1643350(v13);
  v20[2] = a4;
  v20[3] = v9;
  v20[1] = sub_159C470(v14, a3, 0);
  return sub_15E7FB0(a1, 129, (int)v20, 4, v19, 2, a6);
}
