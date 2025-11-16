// Function: sub_2291C00
// Address: 0x2291c00
//
__int64 __fastcall sub_2291C00(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // r15
  unsigned int v10; // r13d
  unsigned int v11; // r8d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // r13d
  unsigned int v18; // edx
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  _QWORD *v23; // [rsp+10h] [rbp-60h]
  int v25; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+38h] [rbp-38h] BYREF

  v7 = sub_2291590(a1, a2, 1, &v26);
  v8 = sub_2291590(a1, a3, 0, v27);
  v9 = sub_2207820(144LL * (unsigned int)(*(_DWORD *)(a1 + 40) + 1));
  v23 = sub_DCC810(*(__int64 **)(a1 + 8), v27[0], v26, 0, 0);
  if ( *(_DWORD *)(a1 + 40) )
  {
    v10 = 1;
    do
    {
      v14 = 32LL * v10;
      v15 = *(_QWORD *)(v7 + v14 + 24);
      if ( !v15 )
        v15 = *(_QWORD *)(v8 + v14 + 24);
      v11 = v10;
      v12 = 144LL * v10++;
      v13 = v9 + v12;
      *(_QWORD *)v13 = v15;
      *(_WORD *)(v13 + 136) = 7;
      sub_2290D50(a1, v7, v8, v9, v11);
    }
    while ( *(_DWORD *)(a1 + 40) >= v10 );
  }
  v16 = sub_2291860(a1, 7, 0, v9, (__int64)v23);
  if ( (_BYTE)v16 )
  {
    v25 = 0;
    if ( (unsigned int)sub_22918F0(a1, 1u, v7, v8, v9, a4, &v25, (__int64)v23) )
    {
      v18 = 1;
      if ( *(_DWORD *)(a1 + 32) )
      {
        while ( 1 )
        {
          v21 = *a4;
          v19 = (*a4 & 1) != 0
              ? (((v21 >> 1) & ~(-1LL << (*a4 >> 58))) >> v18) & 1
              : (*(_QWORD *)(*(_QWORD *)v21 + 8LL * (v18 >> 6)) >> v18) & 1LL;
          if ( (_BYTE)v19 )
          {
            v20 = 16LL * (v18 - 1);
            *(_BYTE *)(v20 + *(_QWORD *)(a5 + 48)) = *(_BYTE *)(v9 + 144LL * v18 + 137)
                                                   & *(_BYTE *)(v20 + *(_QWORD *)(a5 + 48))
                                                   & 7
                                                   | *(_BYTE *)(v20 + *(_QWORD *)(a5 + 48)) & 0xF8;
            if ( (*(_BYTE *)(*(_QWORD *)(a5 + 48) + v20) & 7) == 0 )
              break;
          }
          if ( *(_DWORD *)(a1 + 32) < ++v18 )
            goto LABEL_24;
        }
      }
      else
      {
LABEL_24:
        v16 = 0;
      }
    }
  }
  else
  {
    v16 = 1;
  }
  if ( v9 )
    j_j___libc_free_0_0(v9);
  if ( v7 )
    j_j___libc_free_0_0(v7);
  if ( v8 )
    j_j___libc_free_0_0(v8);
  return v16;
}
