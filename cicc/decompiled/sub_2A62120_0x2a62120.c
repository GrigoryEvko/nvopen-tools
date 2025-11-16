// Function: sub_2A62120
// Address: 0x2a62120
//
bool __fastcall sub_2A62120(char *a1, __int64 a2, char a3, char a4, unsigned int a5)
{
  bool result; // al
  char v10; // dl
  char v11; // r13
  char v12; // al
  unsigned int v13; // eax
  bool v14; // al
  unsigned __int8 v15; // al
  unsigned __int64 v16; // rdi
  const void *v17; // rax
  unsigned __int64 v18; // rdi
  int v19; // eax
  bool v20; // al
  unsigned __int8 v21; // al
  unsigned __int8 v22; // al
  unsigned int v23; // [rsp+8h] [rbp-38h]
  char v24; // [rsp+Fh] [rbp-31h]
  char v25; // [rsp+Fh] [rbp-31h]

  if ( sub_AAF760(a2) )
  {
    result = 0;
    if ( *a1 == 6 )
      return result;
    goto LABEL_3;
  }
  v10 = *a1;
  if ( *a1 == 1 )
    goto LABEL_21;
  if ( v10 != 5 && !a3 )
  {
    v11 = 4;
    v12 = 4;
    if ( v10 == 4 )
      goto LABEL_9;
LABEL_22:
    a1[1] = 0;
    *a1 = v12;
    v19 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a2 + 8) = 0;
    *((_DWORD *)a1 + 4) = v19;
    *((_QWORD *)a1 + 1) = *(_QWORD *)a2;
    *((_DWORD *)a1 + 8) = *(_DWORD *)(a2 + 24);
    *((_QWORD *)a1 + 3) = *(_QWORD *)(a2 + 16);
    *(_DWORD *)(a2 + 24) = 0;
    return 1;
  }
  if ( (unsigned __int8)(v10 - 4) > 1u )
  {
LABEL_21:
    v12 = 5;
    goto LABEL_22;
  }
  v11 = 5;
LABEL_9:
  v13 = *((_DWORD *)a1 + 4);
  *a1 = v11;
  v23 = v13;
  if ( v13 > 0x40 )
  {
    v24 = v10;
    v14 = sub_C43C50((__int64)(a1 + 8), (const void **)a2);
    v10 = v24;
    if ( !v14 )
    {
      if ( !a4 || (v15 = a1[1] + 1, a1[1] = v15, v15 <= a5) )
      {
LABEL_13:
        v16 = *((_QWORD *)a1 + 1);
        if ( v16 )
          j_j___libc_free_0_0(v16);
        goto LABEL_15;
      }
LABEL_3:
      sub_22C0090((unsigned __int8 *)a1);
      *a1 = 6;
      return 1;
    }
LABEL_24:
    if ( *((_DWORD *)a1 + 8) <= 0x40u )
    {
      if ( *((_QWORD *)a1 + 3) != *(_QWORD *)(a2 + 16) )
      {
LABEL_26:
        if ( a4 )
        {
          v21 = a1[1] + 1;
          a1[1] = v21;
          if ( v21 > a5 )
            goto LABEL_3;
        }
        if ( v23 > 0x40 )
          goto LABEL_13;
        goto LABEL_15;
      }
    }
    else
    {
      v25 = v10;
      v20 = sub_C43C50((__int64)(a1 + 24), (const void **)(a2 + 16));
      v10 = v25;
      if ( !v20 )
        goto LABEL_26;
    }
    return v10 != v11;
  }
  v17 = *(const void **)a2;
  if ( *((_QWORD *)a1 + 1) == *(_QWORD *)a2 )
    goto LABEL_24;
  if ( !a4 )
    goto LABEL_16;
  v22 = a1[1] + 1;
  a1[1] = v22;
  if ( a5 < v22 )
    goto LABEL_3;
LABEL_15:
  v17 = *(const void **)a2;
LABEL_16:
  *((_QWORD *)a1 + 1) = v17;
  *((_DWORD *)a1 + 4) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  if ( *((_DWORD *)a1 + 8) > 0x40u )
  {
    v18 = *((_QWORD *)a1 + 3);
    if ( v18 )
      j_j___libc_free_0_0(v18);
  }
  *((_QWORD *)a1 + 3) = *(_QWORD *)(a2 + 16);
  *((_DWORD *)a1 + 8) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 24) = 0;
  return 1;
}
