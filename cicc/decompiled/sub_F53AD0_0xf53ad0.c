// Function: sub_F53AD0
// Address: 0xf53ad0
//
__int64 __fastcall sub_F53AD0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  unsigned int v9; // edx
  int v10; // eax
  int v11; // r13d
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r13
  int v20; // r13d
  __int64 v21; // rax

  v8 = *((_QWORD *)a1 - 4);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 > 0x40 )
      return 0;
    v10 = *a1;
    v11 = v10 - 29;
    if ( v9 )
    {
      v12 = (__int64)(*(_QWORD *)(v8 + 24) << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
      v13 = v12;
      if ( (((_BYTE)v10 - 42) & 0xFD) == 0 )
      {
LABEL_5:
        if ( v10 != 42 )
          v12 = -v12;
        sub_AF6280(a3, v12);
        return *((_QWORD *)a1 - 8);
      }
    }
    else
    {
      v13 = 0;
      v12 = 0;
      if ( (((_BYTE)v10 - 42) & 0xFD) == 0 )
        goto LABEL_5;
    }
    v15 = *(unsigned int *)(a3 + 8);
    if ( v15 + 2 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v15 + 2, 8u, a5, a6);
      v15 = *(unsigned int *)(a3 + 8);
    }
    v16 = *(_QWORD *)a3;
    *(_QWORD *)(v16 + 8 * v15) = 16;
    *(_QWORD *)(v16 + 8 * v15 + 8) = v13;
    *(_DWORD *)(a3 + 8) += 2;
    v19 = sub_F53AB0(v11);
    if ( !v19 )
      return 0;
  }
  else
  {
    v20 = *a1;
    sub_F4FB10(a2, a3, a4, (__int64)a1, a5, a6);
    v19 = sub_F53AB0(v20 - 29);
    if ( !v19 )
      return 0;
  }
  v21 = *(unsigned int *)(a3 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v21 + 1, 8u, v17, v18);
    v21 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v21) = v19;
  ++*(_DWORD *)(a3 + 8);
  return *((_QWORD *)a1 - 8);
}
