// Function: sub_2B097B0
// Address: 0x2b097b0
//
__int64 __fastcall sub_2B097B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        int *a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v11; // r15d
  char v13; // al
  __int64 v14; // rax
  unsigned int v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+Ch] [rbp-44h]
  unsigned int v17; // [rsp+18h] [rbp-38h] BYREF
  int v18[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v17 = 0;
  if ( (_DWORD)a2 != 6 )
    return sub_DFBC30(a1, a2, a3, (__int64)a4, a5, a6, 0, 0, a7, a8, 0);
  if ( a5 <= 2 )
  {
    v11 = 0;
    return sub_DFBC30(a1, 6, a3, (__int64)a4, a5, a6, v11, 0, a7, a8, 0);
  }
  v15 = a6;
  v16 = *(_DWORD *)(a3 + 32);
  v13 = sub_B4F0B0(a4, a5, v16, v18, (int *)&v17);
  v11 = v17;
  a6 = v15;
  if ( !v13 || v16 >= (int)(v17 + v18[0]) || (int)(v17 + v16) > (int)a5 )
    return sub_DFBC30(a1, 6, a3, (__int64)a4, a5, a6, v11, 0, a7, a8, 0);
  v14 = sub_2B08680(*(_QWORD *)(a3 + 24), a5);
  return sub_DFBC30(a1, 4, v14, (__int64)a4, a5, 0, v11, a3, 0, 0, 0);
}
