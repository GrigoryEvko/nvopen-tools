// Function: sub_2B18C70
// Address: 0x2b18c70
//
__int64 __fastcall sub_2B18C70(char *a1, unsigned int a2)
{
  char v2; // al
  unsigned int *v3; // rax
  __int64 v4; // rdx
  unsigned int *i; // r8
  char v6; // cl
  unsigned int v7; // edi
  int v8; // esi
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned __int64 v12; // r14
  unsigned int v13; // r13d
  int v14; // ebx
  unsigned __int64 v15; // rax
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rbx
  unsigned __int64 v19; // r14
  unsigned int v20; // r12d
  int v21; // r13d
  __int64 v23; // [rsp+18h] [rbp-28h]

  v2 = *a1;
  if ( *a1 == 91 )
  {
    v17 = *((_QWORD *)a1 + 1);
    if ( *(_BYTE *)(v17 + 8) != 17 )
      goto LABEL_19;
    v18 = *((_QWORD *)a1 - 4);
    if ( *(_BYTE *)v18 != 17 )
      goto LABEL_19;
    v19 = *(unsigned int *)(v17 + 32);
    v20 = *(_DWORD *)(v18 + 32);
    v21 = *(_DWORD *)(v17 + 32);
    if ( v20 > 0x40 )
    {
      if ( v20 - (unsigned int)sub_C444A0(v18 + 24) > 0x40 )
        goto LABEL_19;
      v15 = **(_QWORD **)(v18 + 24);
      if ( v19 <= v15 )
        goto LABEL_19;
    }
    else
    {
      v15 = *(_QWORD *)(v18 + 24);
      if ( v19 <= v15 )
        goto LABEL_19;
    }
    v16 = v21 * a2;
  }
  else
  {
    if ( v2 != 90 )
    {
      if ( v2 == 94 )
      {
        v3 = (unsigned int *)*((_QWORD *)a1 + 9);
        v4 = *((_QWORD *)a1 + 1);
        for ( i = &v3[*((unsigned int *)a1 + 20)]; i != v3; a2 = v7 + v8 )
        {
          v6 = *(_BYTE *)(v4 + 8);
          v7 = *v3;
          if ( v6 == 15 )
          {
            v8 = *(_DWORD *)(v4 + 12) * a2;
            v4 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL * v7);
          }
          else
          {
            if ( v6 != 16 )
              goto LABEL_19;
            v8 = *(_DWORD *)(v4 + 32) * a2;
            v4 = *(_QWORD *)(v4 + 24);
          }
          ++v3;
        }
        v9 = 1;
        goto LABEL_20;
      }
LABEL_19:
      v9 = 0;
LABEL_20:
      LODWORD(v23) = a2;
      BYTE4(v23) = v9;
      return v23;
    }
    v10 = *((_QWORD *)a1 + 1);
    if ( *(_BYTE *)(v10 + 8) != 17 )
      goto LABEL_19;
    v11 = *(_QWORD *)a1;
    if ( **(_BYTE **)a1 != 17 )
      goto LABEL_19;
    v12 = *(unsigned int *)(v10 + 32);
    v13 = *(_DWORD *)(v11 + 32);
    v14 = *(_DWORD *)(v10 + 32);
    if ( v13 > 0x40 )
    {
      if ( v13 - (unsigned int)sub_C444A0(v11 + 24) > 0x40 )
        goto LABEL_19;
      v15 = **(_QWORD **)(v11 + 24);
      if ( v12 <= v15 )
        goto LABEL_19;
    }
    else
    {
      v15 = *(_QWORD *)(v11 + 24);
      if ( v12 <= v15 )
        goto LABEL_19;
    }
    v16 = v14 * a2;
  }
  BYTE4(v23) = 1;
  LODWORD(v23) = v15 + v16;
  return v23;
}
