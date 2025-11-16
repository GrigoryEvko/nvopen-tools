// Function: sub_37C57D0
// Address: 0x37c57d0
//
__int64 __fastcall sub_37C57D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        int a10,
        int a11,
        char a12)
{
  __int64 v13; // r14
  int v14; // r15d
  char v15; // al
  __int64 v16; // r8
  __int64 v17; // rbx
  unsigned int v18; // esi
  int v19; // eax
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // [rsp+8h] [rbp-48h] BYREF
  __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v27[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a12 )
    return sub_37C5390(a1, (int *)&a7);
  if ( a7 != unk_5051170 )
  {
    v13 = a1 + 32;
    v14 = *(_DWORD *)(a1 + 8);
    v25 = a7;
    v15 = sub_37BD550(a1 + 32, &v25, &v26);
    v17 = v26;
    if ( v15 )
      return *(unsigned int *)(v17 + 8);
    v18 = *(_DWORD *)(a1 + 56);
    v19 = *(_DWORD *)(a1 + 48);
    v27[0] = v26;
    ++*(_QWORD *)(a1 + 32);
    v20 = v19 + 1;
    v21 = 2 * v18;
    if ( 4 * v20 >= 3 * v18 )
    {
      v18 *= 2;
    }
    else if ( v18 - *(_DWORD *)(a1 + 52) - v20 > v18 >> 3 )
    {
LABEL_9:
      *(_DWORD *)(a1 + 48) = v20;
      if ( unk_5051170 != *(_QWORD *)v17 )
        --*(_DWORD *)(a1 + 52);
      v22 = v25;
      *(_DWORD *)(v17 + 8) = 2 * v14;
      *(_QWORD *)v17 = v22;
      v23 = *(unsigned int *)(a1 + 8);
      v24 = v25;
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v23 + 1, 8u, v16, v21);
        v23 = *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v23) = v24;
      ++*(_DWORD *)(a1 + 8);
      return *(unsigned int *)(v17 + 8);
    }
    sub_37C5570(v13, v18);
    sub_37BD550(v13, &v25, v27);
    v17 = v27[0];
    v20 = *(_DWORD *)(a1 + 48) + 1;
    goto LABEL_9;
  }
  return dword_5051178[0];
}
