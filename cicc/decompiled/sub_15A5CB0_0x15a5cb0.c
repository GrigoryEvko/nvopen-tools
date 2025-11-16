// Function: sub_15A5CB0
// Address: 0x15a5cb0
//
__int64 __fastcall sub_15A5CB0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        int a10,
        __int64 a11)
{
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // r12
  int v20; // edx
  __int64 v21; // rdx
  int v22; // eax
  __int64 v26; // [rsp+8h] [rbp-48h]
  _BYTE v27[56]; // [rsp+18h] [rbp-38h] BYREF

  v13 = (int)a2;
  v14 = sub_1644900(*(_QWORD *)(a1 + 8), 64);
  v15 = sub_159C470(v14, a9, 0);
  v18 = sub_1624210(v15, a9, v16, v17);
  if ( a2 && *a2 == 16 )
    v13 = 0;
  v27[4] = 0;
  v19 = *(_QWORD *)(a1 + 8);
  v20 = 0;
  if ( a4 )
  {
    v21 = a4;
    v26 = v18;
    v22 = sub_161FF10(v19, a3, v21);
    v18 = v26;
    v20 = v22;
  }
  return sub_15BD310(v19, 13, v20, a5, a6, v13, a11, a7, 0, a8, (__int64)v27, a10 | 0x80000u, v18, 0, 1);
}
