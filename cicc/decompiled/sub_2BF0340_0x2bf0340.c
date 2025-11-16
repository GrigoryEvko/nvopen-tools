// Function: sub_2BF0340
// Address: 0x2bf0340
//
__int64 __fastcall sub_2BF0340(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r13
  unsigned __int64 v15; // [rsp+8h] [rbp-28h]

  *(_BYTE *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a4;
  *(_QWORD *)a1 = &unk_4A239A8;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  result = 0x100000000LL;
  *(_QWORD *)(a1 + 24) = 0x100000000LL;
  if ( a4 )
  {
    result = *(_QWORD *)(a4 + 16);
    v8 = result & 0xFFFFFFFFFFFFFFF8LL;
    if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( (result & 4) == 0 )
      {
        v9 = sub_22077B0(0x30u);
        if ( v9 )
        {
          *(_QWORD *)v9 = v9 + 16;
          *(_QWORD *)(v9 + 8) = 0x400000000LL;
        }
        v10 = v9;
        v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        v12 = *(unsigned int *)(v11 + 12);
        *(_QWORD *)(a4 + 16) = v10 | 4;
        v13 = *(unsigned int *)(v11 + 8);
        a5 = v13 + 1;
        if ( v13 + 1 > v12 )
        {
          v15 = v11;
          sub_C8D5F0(v11, (const void *)(v11 + 16), v13 + 1, 8u, a5, a6);
          v11 = v15;
          v13 = *(unsigned int *)(v15 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v11 + 8 * v13) = v8;
        v14 = *(_QWORD *)(a4 + 16);
        ++*(_DWORD *)(v11 + 8);
        v8 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      }
      result = *(unsigned int *)(v8 + 8);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
      {
        sub_C8D5F0(v8, (const void *)(v8 + 16), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(v8 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v8 + 8 * result) = a1;
      ++*(_DWORD *)(v8 + 8);
    }
    else
    {
      *(_QWORD *)(a4 + 16) = a1 & 0xFFFFFFFFFFFFFFFBLL;
    }
  }
  return result;
}
