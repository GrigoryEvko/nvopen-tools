// Function: sub_15A6D60
// Address: 0x15a6d60
//
__int64 __fastcall sub_15A6D60(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        char a13)
{
  __int64 v13; // r10
  int v17; // ebx
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // edx
  int v22; // eax
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v27; // [rsp+8h] [rbp-48h]

  v13 = a3;
  v17 = (int)a2;
  if ( a2 && *a2 == 16 )
    v17 = 0;
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  if ( a12 )
  {
    v20 = sub_161FF10(v18, a11, a12);
    v13 = a3;
    v19 = v20;
  }
  v21 = 0;
  if ( a4 )
  {
    v27 = v19;
    v22 = sub_161FF10(v18, v13, a4);
    v19 = v27;
    v21 = v22;
  }
  v23 = sub_15BDB40(
          v18,
          4,
          v21,
          a5,
          a6,
          v17,
          a10,
          a7,
          a8,
          0,
          (unsigned __int8)(a13 != 0) << 24,
          a9,
          0,
          0,
          0,
          v19,
          0,
          0,
          1);
  v24 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)v24 >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, a1 + 64, 0, 8);
    v24 = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v24) = v23;
  ++*(_DWORD *)(a1 + 56);
  sub_15A6B80(a1, v23);
  return v23;
}
