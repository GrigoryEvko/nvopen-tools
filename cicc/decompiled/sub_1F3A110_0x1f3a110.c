// Function: sub_1F3A110
// Address: 0x1f3a110
//
__int64 __fastcall sub_1F3A110(__int64 a1, __int64 a2, unsigned int a3, unsigned int *a4, unsigned int *a5, __int64 a6)
{
  __int64 v9; // rbx
  __int64 (*v10)(void); // rax
  __int64 result; // rax
  unsigned int v12; // eax
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]
  unsigned int v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v9 = 0;
  v10 = *(__int64 (**)(void))(**(_QWORD **)(a6 + 16) + 112LL);
  if ( v10 != sub_1D00B10 )
  {
    v17 = a6;
    v14 = v10();
    a6 = v17;
    v9 = v14;
  }
  if ( a3 )
  {
    v15 = a6;
    v12 = sub_38D70C0(v9 + 8, a3);
    if ( (v12 & 7) != 0 )
      return 0;
    v16 = v12;
    v13 = sub_38D70D0(v9 + 8, a3);
    if ( v13 < 0 || (v13 & 7) != 0 )
    {
      return 0;
    }
    else
    {
      *a4 = v16 >> 3;
      *a5 = (unsigned int)v13 >> 3;
      result = *(unsigned __int8 *)sub_1E0A0C0(v15);
      if ( (_BYTE)result )
        *a5 = (*(_DWORD *)(*(_QWORD *)(v9 + 280)
                         + 24LL
                         * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
                          + *(_DWORD *)(v9 + 288)
                          * (unsigned int)((__int64)(*(_QWORD *)(v9 + 264) - *(_QWORD *)(v9 + 256)) >> 3))
                         + 4) >> 3)
            - *a4
            - *a5;
      else
        return 1;
    }
  }
  else
  {
    *a4 = *(_DWORD *)(*(_QWORD *)(v9 + 280)
                    + 24LL
                    * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
                     + *(_DWORD *)(v9 + 288)
                     * (unsigned int)((__int64)(*(_QWORD *)(v9 + 264) - *(_QWORD *)(v9 + 256)) >> 3))
                    + 4) >> 3;
    *a5 = 0;
    return 1;
  }
  return result;
}
