// Function: sub_35286D0
// Address: 0x35286d0
//
__int64 __fastcall sub_35286D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
      v11 = *(_QWORD *)(a1 + 264)
          + 14LL
          * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 208) + 8LL)
                                - 40LL * *(unsigned __int16 *)(*(_QWORD *)v6 + 68LL)
                                + 6);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v12 = *(_QWORD *)(a1 + 264)
            + 14LL
            * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 208) + 8LL)
                                  - 40LL * *(unsigned __int16 *)(*(_QWORD *)v6 + 68LL)
                                  + 6);
        sub_C8D5F0(a3, v10, result + 1, 8u, v11, a6);
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
