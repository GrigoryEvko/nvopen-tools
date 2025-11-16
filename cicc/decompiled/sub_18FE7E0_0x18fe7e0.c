// Function: sub_18FE7E0
// Address: 0x18fe7e0
//
__int64 __fastcall sub_18FE7E0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r12d
  __int64 v6; // r13
  int v8; // r12d
  int v9; // eax
  int v10; // r8d
  __int64 *v11; // rcx
  unsigned int i; // edx
  __int64 v13; // rdi
  __int64 *v14; // rbx
  __int64 v15; // rsi
  bool v16; // r9
  bool v17; // r10
  char v18; // al
  unsigned int v19; // edx
  int v20; // [rsp+0h] [rbp-40h]
  unsigned int v21; // [rsp+4h] [rbp-3Ch]
  __int64 *v22; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v8 = v4 - 1;
  v9 = sub_18FE780(*a2);
  v10 = 1;
  v11 = 0;
  for ( i = v8 & v9; ; i = v8 & v19 )
  {
    v13 = *a2;
    v14 = (__int64 *)(v6 + 16LL * i);
    v15 = *v14;
    v16 = *v14 == -8;
    v17 = *v14 == -16;
    if ( v16 || *a2 == -8 || *a2 == -16 || *v14 == -16 )
    {
      if ( v13 == v15 )
        goto LABEL_7;
    }
    else
    {
      v20 = v10;
      v21 = i;
      v22 = v11;
      v18 = sub_15F41F0(v13, v15);
      v11 = v22;
      i = v21;
      v10 = v20;
      if ( v18 )
      {
LABEL_7:
        *a3 = v14;
        return 1;
      }
      v15 = *v14;
      v17 = *v14 == -16;
      v16 = *v14 == -8;
    }
    if ( v16 || v17 )
      break;
LABEL_12:
    v19 = v10 + i;
    ++v10;
  }
  if ( v15 != -8 )
  {
    if ( !v11 && v17 )
      v11 = v14;
    goto LABEL_12;
  }
  if ( !v11 )
    v11 = v14;
  *a3 = v11;
  return 0;
}
