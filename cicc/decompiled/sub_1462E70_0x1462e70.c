// Function: sub_1462E70
// Address: 0x1462e70
//
void __fastcall sub_1462E70(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 **v12; // r14
  __int64 v13; // rbx
  int v14; // edx
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r9

  v10 = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_1462D40(a1, a2, a3, v10, a5, a6, a7, a8, a9, a10);
  }
  else
  {
    v11 = v10 >> 4;
    v12 = (__int64 **)&a1[v11];
    v13 = (8 * v11) >> 3;
    sub_1462E70((_DWORD)a1, (_DWORD)a1 + 8 * v11, a3, v11, a5, a6, (__int64)a7, (__int64)a8, (__int64)a9, a10);
    sub_1462E70((_DWORD)v12, (_DWORD)a2, v14, v15, v16, v17, (__int64)a7, (__int64)a8, (__int64)a9, a10);
    sub_14628F0(a1, v12, (__int64)a2, v13, a2 - v12, v18, a7, a8, a9, a10);
  }
}
