// Function: sub_2FDD350
// Address: 0x2fdd350
//
__int64 __fastcall sub_2FDD350(__int64 a1, __int64 a2, unsigned int a3, unsigned int *a4, unsigned int *a5, __int64 a6)
{
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 result; // rax
  unsigned int v13; // eax
  int v14; // eax
  unsigned int v15; // [rsp+4h] [rbp-3Ch]

  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a6 + 16) + 200LL))(*(_QWORD *)(a6 + 16));
  v11 = v10;
  if ( a3 )
  {
    v13 = sub_2FF7530(v10, a3);
    if ( (v13 & 7) != 0 )
      return 0;
    v15 = v13;
    v14 = sub_2FF7550(v11, a3);
    if ( v14 < 0 || (v14 & 7) != 0 )
    {
      return 0;
    }
    else
    {
      *a4 = v15 >> 3;
      *a5 = (unsigned int)v14 >> 3;
      result = *(unsigned __int8 *)sub_2E79000((__int64 *)a6);
      if ( (_BYTE)result )
        *a5 = (*(_DWORD *)(*(_QWORD *)(v11 + 312)
                         + 16LL
                         * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
                          + *(_DWORD *)(v11 + 328)
                          * (unsigned int)((__int64)(*(_QWORD *)(v11 + 288) - *(_QWORD *)(v11 + 280)) >> 3))
                         + 4) >> 3)
            - *a4
            - *a5;
      else
        return 1;
    }
  }
  else
  {
    *a4 = *(_DWORD *)(*(_QWORD *)(v10 + 312)
                    + 16LL
                    * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
                     + *(_DWORD *)(v10 + 328)
                     * (unsigned int)((__int64)(*(_QWORD *)(v10 + 288) - *(_QWORD *)(v10 + 280)) >> 3))
                    + 4) >> 3;
    *a5 = 0;
    return 1;
  }
  return result;
}
