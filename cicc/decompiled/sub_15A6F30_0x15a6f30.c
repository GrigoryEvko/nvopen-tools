// Function: sub_15A6F30
// Address: 0x15a6f30
//
__int64 __fastcall sub_15A6F30(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        _BYTE *a5,
        int a6,
        int a7,
        int a8,
        __int64 a9,
        int a10,
        int a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v13; // r10
  int v17; // ebx
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // edx
  int v22; // eax
  __int64 v23; // r12
  __int64 v26; // [rsp+0h] [rbp-40h]

  v13 = a3;
  v17 = (int)a5;
  if ( a5 && *a5 == 16 )
    v17 = 0;
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  if ( a13 )
  {
    v20 = sub_161FF10(v18, a12, a13);
    v13 = a3;
    v19 = v20;
  }
  v21 = 0;
  if ( a4 )
  {
    v26 = v19;
    v22 = sub_161FF10(v18, v13, a4);
    v19 = v26;
    v21 = v22;
  }
  v23 = sub_15BDB40(v18, a2, v21, a6, a7, v17, 0, a9, a10, 0, a11, 0, a8, 0, 0, v19, 0, 2, 1);
  sub_15A6B80(a1, v23);
  return v23;
}
