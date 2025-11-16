// Function: sub_80B290
// Address: 0x80b290
//
unsigned __int8 *__fastcall sub_80B290(__int64 a1, int a2, __int64 a3)
{
  unsigned __int8 *v3; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  char *v8; // rdx
  char *v9; // rcx
  char v10; // al
  char *v11; // rax
  char *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax

  v3 = 0;
  if ( !*(_DWORD *)(a3 + 48) )
  {
    v6 = qword_4F18BE0;
    ++*(_QWORD *)a3;
    v7 = *(_QWORD *)(v6 + 16);
    if ( (unsigned __int64)(v7 + 1) > *(_QWORD *)(v6 + 8) )
    {
      sub_823810(v6);
      v6 = qword_4F18BE0;
      v7 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(*(_QWORD *)(v6 + 32) + v7) = 0;
    ++*(_QWORD *)(v6 + 16);
    if ( *(_QWORD *)(a3 + 8) )
    {
      v8 = *(char **)(v6 + 32);
      v9 = v8;
      while ( 1 )
      {
        v10 = *v8++;
        if ( v10 != 32 )
          break;
LABEL_9:
        --*(_QWORD *)(v6 + 16);
        --*(_QWORD *)(a3 + 8);
      }
      while ( 1 )
      {
        *v9 = v10;
        if ( !v10 )
          break;
        v10 = *v8++;
        ++v9;
        if ( v10 == 32 )
          goto LABEL_9;
      }
    }
    v3 = *(unsigned __int8 **)(v6 + 32);
    if ( unk_4D042C8 && a2 )
    {
      v3 = sub_809930(*(unsigned __int8 **)(v6 + 32), a1, a3);
      if ( !a1 )
        goto LABEL_18;
    }
    else if ( !a1 )
    {
      goto LABEL_18;
    }
    v11 = (char *)sub_7E1510(*(_QWORD *)a3);
    v12 = strcpy(v11, (const char *)v3);
    if ( (*(_BYTE *)(a1 + 89) & 8) == 0 )
      *(_QWORD *)(a1 + 24) = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a3 + 68) )
      *(_QWORD *)(a1 + 16) = v12;
    else
      *(_QWORD *)(a1 + 8) = v12;
    *(_BYTE *)(a1 + 89) = *(_BYTE *)(a1 + 89) & 0xD7 | (32 * (a2 == 0 && unk_4D042C8 != 0)) | 8;
  }
LABEL_18:
  v13 = *(_QWORD **)(a3 + 16);
  if ( v13 )
  {
    do
    {
      v14 = v13[2];
      qword_4F18C00[BYTE1(v14)] = 0;
      *(_BYTE *)(v14 + 91) &= ~2u;
      v13 = (_QWORD *)*v13;
    }
    while ( v13 );
    **(_QWORD **)(a3 + 24) = qword_4F18BC8;
    qword_4F18BC8 = *(_QWORD *)(a3 + 16);
  }
  v15 = qword_4F18BE8;
  v16 = *(_QWORD *)qword_4F18BF0;
  qword_4F18BE8 = qword_4F18BF0;
  *(_QWORD *)qword_4F18BF0 = v15;
  qword_4F18BF0 = v16;
  if ( v16 )
    v16 = *(_QWORD *)(v16 + 8);
  qword_4F18BE0 = v16;
  return v3;
}
