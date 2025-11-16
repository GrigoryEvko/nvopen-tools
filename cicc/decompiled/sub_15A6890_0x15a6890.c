// Function: sub_15A6890
// Address: 0x15a6890
//
__int64 __fastcall sub_15A6890(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        unsigned __int8 a10,
        __int64 a11,
        __int64 a12,
        int a13)
{
  __int64 v13; // r10
  int v15; // ecx
  __int64 v16; // r14
  int v17; // eax
  int v18; // edx
  int v19; // eax
  __int64 v20; // r12
  __int64 v21; // r12
  __int64 v22; // rax
  int v25; // [rsp+8h] [rbp-38h]

  v13 = a3;
  v15 = 0;
  v16 = *(_QWORD *)(a1 + 8);
  if ( a6 )
  {
    v17 = sub_161FF10(*(_QWORD *)(a1 + 8), a5, a6);
    v13 = a3;
    v15 = v17;
  }
  v18 = 0;
  if ( a4 )
  {
    v25 = v15;
    v19 = sub_161FF10(v16, v13, a4);
    v15 = v25;
    v18 = v19;
  }
  v20 = sub_15C2FB0(v16, a2, v18, v15, a7, a8, a9, a10, 1, a12, a13, 1, 1);
  if ( !a11 )
    a11 = sub_15A6870(a1, 0, 0);
  v21 = sub_15C5570(*(_QWORD *)(a1 + 8), v20, a11, 0, 1);
  v22 = *(unsigned int *)(a1 + 200);
  if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 204) )
  {
    sub_16CD150(a1 + 192, a1 + 208, 0, 8);
    v22 = *(unsigned int *)(a1 + 200);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v22) = v21;
  ++*(_DWORD *)(a1 + 200);
  return v21;
}
