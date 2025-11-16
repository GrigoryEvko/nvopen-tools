// Function: sub_F3A590
// Address: 0xf3a590
//
__int64 __fastcall sub_F3A590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r12
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  unsigned __int8 v15; // cl
  _QWORD *v16; // rsi
  _QWORD *v17; // rcx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  _QWORD *v22; // rcx
  _QWORD *v23; // rcx
  __int64 *v24; // rax
  int v25; // eax
  __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int8 v27; // [rsp+17h] [rbp-39h]

  v4 = *(unsigned int *)(a1 + 20);
  if ( (_DWORD)v4 != *(_DWORD *)(a1 + 24) )
  {
    v27 = 0;
    v26 = a2 + 56;
    while ( 1 )
    {
      v7 = *(__int64 **)(a1 + 8);
      if ( !*(_BYTE *)(a1 + 28) )
        v4 = *(unsigned int *)(a1 + 16);
      v8 = &v7[v4];
      v9 = *v7;
      if ( v7 != v8 )
      {
        while ( 1 )
        {
          v9 = *v7;
          v10 = v7;
          if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v8 == ++v7 )
          {
            v9 = v10[1];
            break;
          }
        }
      }
      v11 = sub_AA56F0(v9);
      v12 = v11;
      if ( !v11 )
        break;
      if ( a2 )
      {
        if ( *(_BYTE *)(a2 + 84) )
        {
          v13 = *(_QWORD **)(a2 + 64);
          v14 = &v13[*(unsigned int *)(a2 + 76)];
          if ( v13 == v14 )
            break;
          while ( v12 != *v13 )
          {
            if ( v14 == ++v13 )
              goto LABEL_26;
          }
        }
        else if ( !sub_C8CA60(v26, v11) )
        {
          break;
        }
      }
      sub_AA5510(v12);
      v15 = sub_F39690(v12, a3, a4, 0, 0, 0, 0);
      if ( v15 )
      {
        v27 = *(_BYTE *)(a1 + 28);
        if ( !v27 )
        {
          v27 = v15;
          v24 = sub_C8CA60(a1, v12);
          if ( !v24 )
          {
LABEL_42:
            v20 = *(_DWORD *)(a1 + 24);
            v4 = *(unsigned int *)(a1 + 20);
            goto LABEL_23;
          }
          goto LABEL_40;
        }
        v4 = *(unsigned int *)(a1 + 20);
        v16 = *(_QWORD **)(a1 + 8);
        v17 = &v16[v4];
        v18 = v16;
        if ( v16 == v17 )
          goto LABEL_32;
        while ( v12 != *v18 )
        {
          if ( v17 == ++v18 )
            goto LABEL_32;
        }
      }
      else
      {
        if ( !*(_BYTE *)(a1 + 28) )
        {
LABEL_39:
          v24 = sub_C8CA60(a1, v9);
          if ( !v24 )
            goto LABEL_42;
LABEL_40:
          *v24 = -2;
          v25 = *(_DWORD *)(a1 + 24);
          ++*(_QWORD *)a1;
          v4 = *(unsigned int *)(a1 + 20);
          v20 = v25 + 1;
          *(_DWORD *)(a1 + 24) = v20;
          goto LABEL_23;
        }
        v4 = *(unsigned int *)(a1 + 20);
        v16 = *(_QWORD **)(a1 + 8);
        v23 = &v16[v4];
        v18 = v16;
        if ( v16 == v23 )
        {
LABEL_32:
          v20 = *(_DWORD *)(a1 + 24);
          goto LABEL_23;
        }
        while ( *v18 != v9 )
        {
          if ( v23 == ++v18 )
            goto LABEL_32;
        }
      }
LABEL_22:
      v19 = (unsigned int)(v4 - 1);
      *(_DWORD *)(a1 + 20) = v19;
      *v18 = v16[v19];
      v20 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)a1;
      v4 = *(unsigned int *)(a1 + 20);
LABEL_23:
      if ( (_DWORD)v4 == v20 )
        return v27;
    }
LABEL_26:
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_39;
    v4 = *(unsigned int *)(a1 + 20);
    v16 = *(_QWORD **)(a1 + 8);
    v22 = &v16[v4];
    if ( v16 == v22 )
      goto LABEL_32;
    v18 = *(_QWORD **)(a1 + 8);
    while ( *v18 != v9 )
    {
      if ( v22 == ++v18 )
        goto LABEL_32;
    }
    goto LABEL_22;
  }
  return 0;
}
