// Function: sub_27B9BA0
// Address: 0x27b9ba0
//
__int64 __fastcall sub_27B9BA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // rdi
  unsigned int v10; // r14d
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  unsigned int v14; // eax
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  char *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  char *v25; // [rsp+8h] [rbp-48h]
  char *v26; // [rsp+10h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 1;
  v5 = a3 - 24;
  v7 = *a1;
  if ( !a3 )
    v5 = 0;
  if ( (unsigned __int8)sub_B19DB0(v7, a2, v5) )
    return 1;
  if ( *(_BYTE *)(a5 + 28) )
  {
    v12 = *(_QWORD **)(a5 + 8);
    v13 = &v12[*(unsigned int *)(a5 + 20)];
    if ( v12 != v13 )
    {
      while ( a2 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_14;
      }
      return 1;
    }
LABEL_14:
    LOBYTE(v14) = sub_991A70((unsigned __int8 *)a2, v5, a1[3], *a1, 0, 1u, 0);
    v10 = v14;
    if ( !(_BYTE)v14 )
      return v10;
    if ( (unsigned __int8)sub_B46420(a2) )
      return 0;
    if ( *(_BYTE *)(a5 + 28) )
    {
      v19 = *(__int64 **)(a5 + 8);
      v16 = *(unsigned int *)(a5 + 20);
      v15 = &v19[v16];
      if ( v19 != v15 )
      {
        while ( a2 != *v19 )
        {
          if ( v15 == ++v19 )
            goto LABEL_34;
        }
LABEL_21:
        v20 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v21 = v20;
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        {
          v22 = *(char **)(a2 - 8);
          v25 = &v22[v20];
        }
        else
        {
          v25 = (char *)a2;
          v22 = (char *)(a2 - v20);
        }
        v23 = v20 >> 5;
        v24 = v21 >> 7;
        if ( v24 )
        {
          v26 = &v22[128 * v24];
          while ( (unsigned __int8)sub_27B9BA0(a1, *(_QWORD *)v22, a3, a4, a5) )
          {
            if ( !(unsigned __int8)sub_27B9BA0(a1, *((_QWORD *)v22 + 4), a3, a4, a5) )
            {
              LOBYTE(v10) = v25 == v22 + 32;
              return v10;
            }
            if ( !(unsigned __int8)sub_27B9BA0(a1, *((_QWORD *)v22 + 8), a3, a4, a5) )
            {
              LOBYTE(v10) = v25 == v22 + 64;
              return v10;
            }
            if ( !(unsigned __int8)sub_27B9BA0(a1, *((_QWORD *)v22 + 12), a3, a4, a5) )
            {
              LOBYTE(v10) = v25 == v22 + 96;
              return v10;
            }
            v22 += 128;
            if ( v26 == v22 )
            {
              v23 = (v25 - v22) >> 5;
              goto LABEL_37;
            }
          }
          goto LABEL_30;
        }
LABEL_37:
        if ( v23 != 2 )
        {
          if ( v23 != 3 )
          {
            if ( v23 != 1 )
              return v10;
            goto LABEL_40;
          }
          if ( !(unsigned __int8)sub_27B9BA0(a1, *(_QWORD *)v22, a3, a4, a5) )
          {
            LOBYTE(v10) = v22 == v25;
            return v10;
          }
          v22 += 32;
        }
        if ( !(unsigned __int8)sub_27B9BA0(a1, *(_QWORD *)v22, a3, a4, a5) )
        {
LABEL_30:
          LOBYTE(v10) = v25 == v22;
          return v10;
        }
        v22 += 32;
LABEL_40:
        v10 = sub_27B9BA0(a1, *(_QWORD *)v22, a3, a4, a5);
        if ( (_BYTE)v10 )
          return v10;
        goto LABEL_30;
      }
LABEL_34:
      if ( (unsigned int)v16 < *(_DWORD *)(a5 + 16) )
      {
        *(_DWORD *)(a5 + 20) = v16 + 1;
        *v15 = a2;
        ++*(_QWORD *)a5;
        goto LABEL_21;
      }
    }
    sub_C8CC70(a5, a2, (__int64)v15, v16, v17, v18);
    goto LABEL_21;
  }
  if ( !sub_C8CA60(a5, a2) )
    goto LABEL_14;
  return 1;
}
