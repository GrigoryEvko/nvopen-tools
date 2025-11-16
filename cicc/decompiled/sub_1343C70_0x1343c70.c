// Function: sub_1343C70
// Address: 0x1343c70
//
__int64 __fastcall sub_1343C70(
        _BYTE *a1,
        __int64 a2,
        unsigned int *a3,
        unsigned __int64 **a4,
        unsigned __int64 **a5,
        unsigned __int64 **a6,
        unsigned __int64 **a7,
        unsigned __int64 **a8,
        __int64 a9,
        __int64 a10)
{
  unsigned int v10; // r10d
  unsigned __int64 *v14; // rcx
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r14
  unsigned __int64 *v21; // rcx
  unsigned __int64 *v22; // rax
  unsigned __int64 *v23; // rax
  unsigned int *v24; // [rsp+0h] [rbp-40h]

  v10 = 1;
  v14 = *a4;
  v15 = v14[1] & 0xFFFFFFFFFFFFF000LL;
  v16 = v14[2] & 0xFFFFFFFFFFFFF000LL;
  v17 = ((v15 + ((a10 + 4095) & 0xFFFFFFFFFFFFF000LL) - 1) & -(__int64)((a10 + 4095) & 0xFFFFFFFFFFFFF000LL)) - v15;
  if ( v16 >= v17 + a9 )
  {
    *a5 = 0;
    v18 = v16
        + v15
        - ((v15 + ((a10 + 4095) & 0xFFFFFFFFFFFFF000LL) - 1) & -(__int64)((a10 + 4095) & 0xFFFFFFFFFFFFF000LL));
    *a6 = 0;
    v19 = v18 - a9;
    *a7 = 0;
    *a8 = 0;
    if ( !v17
      || (v21 = *a4, v24 = a3, *a5 = *a4, v22 = sub_1343890(a1, a2, a3, v21, v17, v18), a3 = v24, (*a4 = v22) != 0) )
    {
      if ( !v19 )
        return 0;
      v23 = sub_1343890(a1, a2, a3, *a4, a9, v19);
      *a6 = v23;
      if ( v23 )
      {
        return 0;
      }
      else
      {
        v10 = 2;
        *a7 = *a4;
        *a8 = *a5;
        *a5 = 0;
        *a4 = 0;
      }
    }
    else
    {
      v10 = 2;
      *a7 = *a5;
      *a5 = 0;
    }
  }
  return v10;
}
