// Function: sub_2ABDFA0
// Address: 0x2abdfa0
//
__int64 __fastcall sub_2ABDFA0(__int64 a1)
{
  __int64 v1; // r14
  _QWORD *v2; // r13
  __int64 v3; // rdx
  _QWORD *v4; // r15
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int8 v12; // [rsp+7h] [rbp-39h]
  unsigned __int8 *v13; // [rsp+8h] [rbp-38h]

  v1 = 0;
  v2 = *(_QWORD **)(a1 + 48);
  v3 = *(unsigned int *)(a1 + 56);
  v4 = &v2[v3];
  if ( *(_BYTE *)(a1 + 161) )
    v1 = v2[(unsigned int)(v3 - 1)];
  v13 = *(unsigned __int8 **)(a1 + 136);
  v5 = sub_22077B0(0xA8u);
  v6 = v5;
  if ( v5 )
  {
    v12 = *(_BYTE *)(a1 + 160);
    sub_2ABDBC0(v5, 9, v2, v4, v13, v12);
    v8 = v12;
    *(_BYTE *)(v6 + 160) = v12;
    *(_QWORD *)v6 = &unk_4A237B0;
    *(_QWORD *)(v6 + 40) = &unk_4A237F8;
    *(_QWORD *)(v6 + 96) = &unk_4A23830;
    *(_BYTE *)(v6 + 161) = v1 != 0;
    if ( v1 )
    {
      v9 = *(unsigned int *)(v6 + 56);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
      {
        sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v9 + 1, 8u, v7, v12);
        v9 = *(unsigned int *)(v6 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v9) = v1;
      ++*(_DWORD *)(v6 + 56);
      v10 = *(unsigned int *)(v1 + 24);
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
      {
        sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v10 + 1, 8u, v7, v8);
        v10 = *(unsigned int *)(v1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v10) = v6 + 40;
      ++*(_DWORD *)(v1 + 24);
    }
  }
  *(_BYTE *)(v6 + 152) = *(_BYTE *)(a1 + 152);
  *(_DWORD *)(v6 + 156) = *(_DWORD *)(a1 + 156);
  return v6;
}
