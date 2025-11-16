// Function: sub_2B1C6E0
// Address: 0x2b1c6e0
//
void __fastcall sub_2B1C6E0(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // rcx
  __int64 v9; // rcx
  unsigned int *v10; // r14
  __int64 v11; // rbx
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  unsigned int *v16; // r9

  v8 = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 56 )
  {
    sub_2B1C2A0(a1, a2, a3, v8, a5, a6, a7, a8);
  }
  else
  {
    v9 = v8 >> 3;
    v10 = &a1[v9];
    v11 = (4 * v9) >> 2;
    sub_2B1C6E0((_DWORD)a1, (_DWORD)a1 + 4 * v9, a3, v9, a5, a6, a7, *((__int64 *)&a7 + 1), a8);
    sub_2B1C6E0((_DWORD)v10, (_DWORD)a2, v12, v13, v14, v15, a7, *((__int64 *)&a7 + 1), a8);
    sub_2B1C4B0(a1, v10, (__int64)a2, v11, a2 - v10, v16, a7, a8);
  }
}
