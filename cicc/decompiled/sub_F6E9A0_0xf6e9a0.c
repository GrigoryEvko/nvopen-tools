// Function: sub_F6E9A0
// Address: 0xf6e9a0
//
__int64 __fastcall sub_F6E9A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 **v17; // r12
  __int64 **v18; // r13
  __int64 v19; // rbx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  const void *v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  v26 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  v7 = *a3;
  v27 = a4 + 56;
  if ( *(_BYTE *)(a4 + 84) )
  {
    v8 = *(_QWORD **)(a4 + 64);
    v9 = &v8[*(unsigned int *)(a4 + 76)];
    if ( v8 == v9 )
      return a1;
    while ( v7 != *v8 )
    {
      if ( v9 == ++v8 )
        return a1;
    }
    v10 = (__int64 *)(a1 + 16);
    goto LABEL_7;
  }
  if ( sub_C8CA60(v27, v7) )
  {
    v25 = *(unsigned int *)(a1 + 8);
    if ( v25 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v10 = (__int64 *)(*(_QWORD *)a1 + 8 * v25);
    }
    else
    {
      sub_C8D5F0(a1, v26, v25 + 1, 8u, v24, a6);
      v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
LABEL_7:
    *v10 = v7;
    v11 = *(_DWORD *)(a1 + 8) + 1;
    *(_DWORD *)(a1 + 8) = v11;
    goto LABEL_8;
  }
  v11 = *(_DWORD *)(a1 + 8);
LABEL_8:
  if ( v11 )
  {
    v28 = 0;
    while ( 1 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v28);
      if ( v12 )
      {
        v13 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
        v14 = *(_DWORD *)(v12 + 44) + 1;
      }
      else
      {
        v13 = 0;
        v14 = 0;
      }
      if ( v14 >= *(_DWORD *)(a2 + 32) )
        BUG();
      v15 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v13);
      v16 = *(_QWORD *)(v15 + 24);
      v17 = (__int64 **)(v16 + 8LL * *(unsigned int *)(v15 + 32));
      v18 = (__int64 **)v16;
      if ( (__int64 **)v16 != v17 )
        break;
LABEL_23:
      if ( *(unsigned int *)(a1 + 8) <= (unsigned __int64)++v28 )
        return a1;
    }
    while ( 1 )
    {
      v19 = **v18;
      if ( *(_BYTE *)(a4 + 84) )
      {
        v20 = *(_QWORD **)(a4 + 64);
        v21 = &v20[*(unsigned int *)(a4 + 76)];
        if ( v20 != v21 )
        {
          while ( v19 != *v20 )
          {
            if ( v21 == ++v20 )
              goto LABEL_22;
          }
LABEL_19:
          v22 = *(unsigned int *)(a1 + 8);
          if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, v26, v22 + 1, 8u, v16, a6);
            v22 = *(unsigned int *)(a1 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v22) = v19;
          ++*(_DWORD *)(a1 + 8);
        }
      }
      else if ( sub_C8CA60(v27, **v18) )
      {
        goto LABEL_19;
      }
LABEL_22:
      if ( v17 == ++v18 )
        goto LABEL_23;
    }
  }
  return a1;
}
