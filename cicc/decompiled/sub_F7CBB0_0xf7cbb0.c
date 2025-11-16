// Function: sub_F7CBB0
// Address: 0xf7cbb0
//
__int64 __fastcall sub_F7CBB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  _BYTE *v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rax
  _BYTE *v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  const void *v23; // [rsp+0h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  v23 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x2000000000LL;
  if ( *(_DWORD *)(a2 + 80) )
  {
    v7 = *(_QWORD *)(a2 + 72);
    v8 = v7 + 24LL * *(unsigned int *)(a2 + 88);
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v7 + 16);
        if ( v9 != -4096 && v9 != -8192 )
          break;
        v7 += 24;
        if ( v8 == v7 )
          goto LABEL_2;
      }
      if ( v8 != v7 )
      {
        v10 = *(_BYTE **)(v7 + 16);
        if ( !*(_BYTE *)(a2 + 156) )
          goto LABEL_22;
LABEL_11:
        v11 = *(_QWORD **)(a2 + 136);
        v12 = &v11[*(unsigned int *)(a2 + 148)];
        if ( v11 == v12 )
        {
LABEL_23:
          if ( *v10 > 0x1Cu )
          {
            v14 = *(unsigned int *)(a1 + 8);
            if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, v23, v14 + 1, 8u, a5, a6);
              v14 = *(unsigned int *)(a1 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = v10;
            ++*(_DWORD *)(a1 + 8);
          }
          goto LABEL_15;
        }
        while ( v10 != (_BYTE *)*v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_23;
        }
LABEL_15:
        while ( 1 )
        {
          v7 += 24;
          if ( v7 == v8 )
            break;
          while ( 1 )
          {
            v13 = *(_QWORD *)(v7 + 16);
            if ( v13 != -4096 && v13 != -8192 )
              break;
            v7 += 24;
            if ( v8 == v7 )
              goto LABEL_2;
          }
          if ( v8 == v7 )
            break;
          v10 = *(_BYTE **)(v7 + 16);
          if ( *(_BYTE *)(a2 + 156) )
            goto LABEL_11;
LABEL_22:
          if ( !sub_C8CA60(a2 + 128, (__int64)v10) )
            goto LABEL_23;
        }
      }
    }
  }
LABEL_2:
  if ( *(_DWORD *)(a2 + 112) )
  {
    v15 = *(_QWORD *)(a2 + 104);
    v16 = v15 + 24LL * *(unsigned int *)(a2 + 120);
    if ( v15 != v16 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(v15 + 16);
        if ( v17 != -4096 && v17 != -8192 )
          break;
        v15 += 24;
        if ( v16 == v15 )
          return a1;
      }
      if ( v16 != v15 )
      {
        v18 = *(_BYTE **)(v15 + 16);
        if ( !*(_BYTE *)(a2 + 156) )
          goto LABEL_45;
LABEL_34:
        v19 = *(_QWORD **)(a2 + 136);
        v20 = &v19[*(unsigned int *)(a2 + 148)];
        if ( v19 == v20 )
        {
LABEL_46:
          if ( *v18 > 0x1Cu )
          {
            v22 = *(unsigned int *)(a1 + 8);
            if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, v23, v22 + 1, 8u, a5, a6);
              v22 = *(unsigned int *)(a1 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a1 + 8 * v22) = v18;
            ++*(_DWORD *)(a1 + 8);
          }
          goto LABEL_38;
        }
        while ( v18 != (_BYTE *)*v19 )
        {
          if ( v20 == ++v19 )
            goto LABEL_46;
        }
LABEL_38:
        while ( 1 )
        {
          v15 += 24;
          if ( v15 == v16 )
            break;
          while ( 1 )
          {
            v21 = *(_QWORD *)(v15 + 16);
            if ( v21 != -8192 && v21 != -4096 )
              break;
            v15 += 24;
            if ( v16 == v15 )
              return a1;
          }
          if ( v16 == v15 )
            break;
          v18 = *(_BYTE **)(v15 + 16);
          if ( *(_BYTE *)(a2 + 156) )
            goto LABEL_34;
LABEL_45:
          if ( !sub_C8CA60(a2 + 128, (__int64)v18) )
            goto LABEL_46;
        }
      }
    }
  }
  return a1;
}
