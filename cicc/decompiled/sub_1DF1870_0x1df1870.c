// Function: sub_1DF1870
// Address: 0x1df1870
//
__int64 __fastcall sub_1DF1870(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // r14
  const void *v10; // r15
  __int64 v11; // r8
  __int64 v12; // [rsp+8h] [rbp-38h]

  v6 = *a2;
  result = *((unsigned int *)a2 + 2);
  v8 = *a2 + 8 * result;
  if ( *a2 != v8 )
  {
    result = *(unsigned int *)(a3 + 8);
    v10 = (const void *)(a3 + 16);
    do
    {
      v11 = *(_QWORD *)(a1 + 296)
          + 14LL
          * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL)
                                + ((unsigned __int64)**(unsigned __int16 **)(*(_QWORD *)v6 + 16LL) << 6)
                                + 6);
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)result )
      {
        v12 = *(_QWORD *)(a1 + 296)
            + 14LL
            * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL)
                                  + ((unsigned __int64)**(unsigned __int16 **)(*(_QWORD *)v6 + 16LL) << 6)
                                  + 6);
        sub_16CD150(a3, v10, 0, 8, v11, a6);
        result = *(unsigned int *)(a3 + 8);
        v11 = v12;
      }
      v6 += 8;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v11;
      result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = result;
    }
    while ( v8 != v6 );
  }
  return result;
}
