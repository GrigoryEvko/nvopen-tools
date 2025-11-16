// Function: sub_33DFEB0
// Address: 0x33dfeb0
//
__int64 __fastcall sub_33DFEB0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int *v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  int v26; // [rsp+4h] [rbp-4Ch]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]

  if ( !sub_33DFCF0(a1, a2, a5) )
  {
    v13 = sub_33DFBC0(a3, a4, 0, 0, v9, v10);
    if ( !v13 || *(_DWORD *)(a1 + 24) != 215 )
      return 0;
    v14 = *(__int64 **)(a1 + 40);
    v29 = *v14;
    v15 = *v14;
    v28 = v14[1];
    v16 = sub_33C9580(*v14, *((_DWORD *)v14 + 2));
    v17 = *(_QWORD *)(v13 + 96);
    v18 = v16;
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
    {
      v26 = *(_DWORD *)(v17 + 32);
      v27 = v16;
      v25 = sub_C444A0(v17 + 24);
      v18 = v27;
      v20 = (unsigned int)(v26 - v25);
    }
    else
    {
      v19 = *(_QWORD *)(v17 + 24);
      if ( !v19 )
        goto LABEL_11;
      _BitScanReverse64(&v19, v19);
      v20 = 64 - ((unsigned int)v19 ^ 0x3F);
    }
    if ( v18 < v20 )
      return 0;
LABEL_11:
    if ( sub_33DFCF0(v29, v28, a5) )
    {
      v21 = *(_QWORD *)(v15 + 40);
      if ( *(_DWORD *)(*(_QWORD *)v21 + 24LL) == 216 )
      {
        v22 = *(unsigned int **)(*(_QWORD *)v21 + 40LL);
        v23 = *(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2;
        v24 = *(_QWORD *)(*(_QWORD *)v22 + 48LL) + 16LL * v22[2];
        if ( *(_WORD *)v24 == *(_WORD *)v23 && (*(_QWORD *)(v24 + 8) == *(_QWORD *)(v23 + 8) || *(_WORD *)v24) )
          return *(_QWORD *)v22;
      }
    }
    return 0;
  }
  return **(_QWORD **)(a1 + 40);
}
