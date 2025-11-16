// Function: sub_3200960
// Address: 0x3200960
//
__int64 __fastcall sub_3200960(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  unsigned int *v6; // r12
  unsigned int v7; // edx
  unsigned int *v8; // rax
  __int64 v10; // rax
  char v11; // dl
  unsigned int v12; // r13d
  unsigned int *v13; // rdx
  __int64 v14; // rax
  unsigned int *v15; // r14
  __int64 i; // rax
  __int64 v17; // rdx
  bool v18; // r10
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // [rsp+4h] [rbp-4Ch]
  _QWORD *v22; // [rsp+8h] [rbp-48h]
  _QWORD *v23; // [rsp+10h] [rbp-40h]
  _QWORD *v24; // [rsp+18h] [rbp-38h]

  if ( *(_QWORD *)(a2 + 64) )
  {
    v10 = sub_31FE900(a2 + 24, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v10;
    *(_BYTE *)(a1 + 16) = v11;
    return a1;
  }
  v6 = (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8));
  if ( *(unsigned int **)a2 != v6 )
  {
    v7 = *a3;
    v8 = *(unsigned int **)a2;
    do
    {
      if ( *v8 == v7 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v8;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
      ++v8;
    }
    while ( v6 != v8 );
    if ( !*(_DWORD *)(a2 + 8) )
      goto LABEL_10;
    v15 = *(unsigned int **)a2;
    v23 = (_QWORD *)(a2 + 24);
    v24 = (_QWORD *)(a2 + 32);
    for ( i = sub_3200860((_QWORD *)(a2 + 24), a2 + 32, *(unsigned int **)a2); ; i = sub_3200860(v23, (__int64)v24, v15) )
    {
      if ( v17 )
      {
        v18 = i || (_QWORD *)v17 == v24 || *v15 < *(_DWORD *)(v17 + 32);
        v21 = v18;
        v22 = (_QWORD *)v17;
        v19 = sub_22077B0(0x28u);
        *(_DWORD *)(v19 + 32) = *v15;
        sub_220F040(v21, v19, v22, v24);
        ++*(_QWORD *)(a2 + 64);
      }
      if ( v6 == ++v15 )
        break;
    }
LABEL_24:
    *(_DWORD *)(a2 + 8) = 0;
    v20 = sub_31FE900((__int64)v23, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v20;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  if ( *(_DWORD *)(a2 + 8) )
  {
    v23 = (_QWORD *)(a2 + 24);
    goto LABEL_24;
  }
LABEL_10:
  v12 = *a3;
  if ( !*(_DWORD *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), 1u, 4u, a5, *(_QWORD *)a2);
    v6 = (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8));
  }
  *v6 = v12;
  v13 = *(unsigned int **)a2;
  v14 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v14;
  *(_BYTE *)(a1 + 8) = 1;
  *(_QWORD *)a1 = &v13[v14 - 1];
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
