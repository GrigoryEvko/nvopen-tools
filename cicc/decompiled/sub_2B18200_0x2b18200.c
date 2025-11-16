// Function: sub_2B18200
// Address: 0x2b18200
//
__int64 __fastcall sub_2B18200(__int64 a1, unsigned int a2)
{
  __int64 v2; // r15
  __int64 **v4; // r9
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // r10
  __int64 v11; // rbx
  unsigned __int8 **v12; // rcx
  int v13; // eax
  unsigned __int8 **v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // r12
  int v17; // [rsp+Ch] [rbp-74h]
  __int64 **v18; // [rsp+10h] [rbp-70h]
  unsigned __int8 **v19; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+28h] [rbp-58h]
  _BYTE v21[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  if ( *(_BYTE *)v2 == 13 )
    return 0;
  v4 = *(__int64 ***)(*(_QWORD *)(a1 + 16) + 3296LL);
  v5 = 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v6 = **(unsigned int **)(a1 + 24);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
  {
    v7 = *(_QWORD *)(v2 - 8);
    v8 = v7 + v5;
  }
  else
  {
    v7 = v2 - v5;
    v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  }
  v9 = v8 - v7;
  v19 = (unsigned __int8 **)v21;
  v10 = v9 >> 5;
  v20 = 0x400000000LL;
  v11 = v9 >> 5;
  if ( (unsigned __int64)v9 > 0x80 )
  {
    v17 = v6;
    v18 = v4;
    sub_C8D5F0((__int64)&v19, v21, v9 >> 5, 8u, v6, (__int64)v4);
    v14 = v19;
    v13 = v20;
    v10 = v9 >> 5;
    v4 = v18;
    LODWORD(v6) = v17;
    v12 = &v19[(unsigned int)v20];
  }
  else
  {
    v12 = (unsigned __int8 **)v21;
    v13 = 0;
    v14 = (unsigned __int8 **)v21;
  }
  if ( v9 > 0 )
  {
    v15 = 0;
    do
    {
      v12[v15 / 8] = *(unsigned __int8 **)(v7 + 4 * v15);
      v15 += 8LL;
      --v11;
    }
    while ( v11 );
    v14 = v19;
    v13 = v20;
  }
  LODWORD(v20) = v10 + v13;
  v16 = sub_DFCEF0(v4, (unsigned __int8 *)v2, v14, (unsigned int)(v10 + v13), v6);
  if ( v19 != (unsigned __int8 **)v21 )
    _libc_free((unsigned __int64)v19);
  return v16;
}
