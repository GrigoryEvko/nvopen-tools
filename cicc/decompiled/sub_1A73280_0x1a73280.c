// Function: sub_1A73280
// Address: 0x1a73280
//
__int64 __fastcall sub_1A73280(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r15
  int v22; // eax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rbx
  int v26; // r8d
  int v27; // r9d
  __int64 result; // rax
  __int64 v29[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a2;
  v29[0] = a3;
  v4 = sub_157F280(a3);
  v9 = v5;
  v10 = v4;
  while ( v9 != v10 )
  {
    v21 = sub_1599EF0(*(__int64 ***)v10);
    v22 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
    if ( v22 == *(_DWORD *)(v10 + 56) )
    {
      sub_15F55D0(v10, a2, v19, v20, v7, v8);
      v22 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
    }
    v23 = (v22 + 1) & 0xFFFFFFF;
    v24 = v23 | *(_DWORD *)(v10 + 20) & 0xF0000000;
    *(_DWORD *)(v10 + 20) = v24;
    if ( (v24 & 0x40000000) != 0 )
      v11 = *(_QWORD *)(v10 - 8);
    else
      v11 = v10 - 24 * v23;
    v12 = (__int64 *)(v11 + 24LL * (unsigned int)(v23 - 1));
    if ( *v12 )
    {
      v13 = v12[1];
      v14 = v12[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v13;
      if ( v13 )
      {
        a2 = *(_QWORD *)(v13 + 16) & 3LL;
        *(_QWORD *)(v13 + 16) = a2 | v14;
      }
    }
    *v12 = v21;
    if ( v21 )
    {
      v15 = *(_QWORD *)(v21 + 8);
      a2 = v21 + 8;
      v12[1] = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = (unsigned __int64)(v12 + 1) | *(_QWORD *)(v15 + 16) & 3LL;
      v12[2] = a2 | v12[2] & 3;
      *(_QWORD *)(v21 + 8) = v12;
    }
    v16 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
    v17 = (unsigned int)(v16 - 1);
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v6 = *(_QWORD *)(v10 - 8);
    else
      v6 = v10 - 24 * v16;
    v5 = 3LL * *(unsigned int *)(v10 + 56);
    *(_QWORD *)(v6 + 8 * v17 + 24LL * *(unsigned int *)(v10 + 56) + 8) = v3;
    v18 = *(_QWORD *)(v10 + 32);
    if ( !v18 )
      BUG();
    v10 = 0;
    if ( *(_BYTE *)(v18 - 8) == 77 )
      v10 = v18 - 24;
  }
  v25 = sub_1A72F60(a1 + 448, v29, v5, v6, v7, v8);
  result = *(unsigned int *)(v25 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(v25 + 12) )
  {
    sub_16CD150(v25, (const void *)(v25 + 16), 0, 8, v26, v27);
    result = *(unsigned int *)(v25 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v25 + 8 * result) = v3;
  ++*(_DWORD *)(v25 + 8);
  return result;
}
