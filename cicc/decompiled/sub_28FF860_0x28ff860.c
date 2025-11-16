// Function: sub_28FF860
// Address: 0x28ff860
//
unsigned __int8 *__fastcall sub_28FF860(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 *v6; // r12
  int v7; // eax
  const void *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax

  v6 = a2;
  v7 = *a2;
  if ( (unsigned __int8)v7 > 0x1Cu )
  {
    v8 = (const void *)(a1 + 16);
    do
    {
      if ( (_BYTE)v7 == 63 )
      {
        v12 = *(unsigned int *)(a1 + 8);
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v8, v12 + 1, 8u, a5, a6);
          v12 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) = v6;
        ++*(_DWORD *)(a1 + 8);
        v6 = *(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
      }
      else
      {
        if ( (unsigned int)(v7 - 67) > 0xC )
          return v6;
        v9 = sub_B43CC0((__int64)v6);
        if ( !sub_B507F0(v6, v9) )
          return v6;
        v10 = *(unsigned int *)(a1 + 8);
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v8, v10 + 1, 8u, a5, a6);
          v10 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v10) = v6;
        ++*(_DWORD *)(a1 + 8);
        v6 = (unsigned __int8 *)*((_QWORD *)v6 - 4);
      }
      v7 = *v6;
    }
    while ( (unsigned __int8)v7 > 0x1Cu );
  }
  return v6;
}
