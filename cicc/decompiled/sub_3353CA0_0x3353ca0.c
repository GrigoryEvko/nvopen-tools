// Function: sub_3353CA0
// Address: 0x3353ca0
//
__int64 __fastcall sub_3353CA0(_QWORD *a1)
{
  __int64 *v1; // rax
  __int64 *v2; // r9
  __int64 v3; // rdx
  __int64 v5; // rdi
  __int64 v6; // r8
  unsigned int v7; // r14d
  unsigned int v8; // ebx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 *v11; // rcx
  __int64 v12; // rsi
  unsigned __int8 v13; // dl
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // edx
  int v18; // r11d
  bool v20; // al
  int v21; // [rsp-3Ch] [rbp-3Ch]

  v1 = (__int64 *)a1[3];
  v2 = (__int64 *)a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = (char *)v1 - (char *)v2;
  v5 = *v2;
  if ( (unsigned __int64)((char *)v1 - (char *)v2) > 0x1F40 )
  {
    LODWORD(v6) = 1000;
  }
  else
  {
    v6 = v3 >> 3;
    if ( v3 == 8 )
      goto LABEL_17;
  }
  v7 = 0;
  v8 = 1;
  do
  {
    while ( 1 )
    {
      v9 = v8;
      v10 = v7;
      v11 = &v2[v9];
      v12 = v2[v9];
      v13 = (*(_BYTE *)(v5 + 249) & 0x10) != 0;
      v14 = (*(_BYTE *)(v12 + 249) & 0x10) != 0;
      if ( v13 != v14 )
        break;
      v15 = *(_QWORD *)v12;
      if ( *(_QWORD *)v5 )
      {
        v16 = *(_DWORD *)(*(_QWORD *)v5 + 72LL);
        if ( !v15 )
        {
          v18 = *(_DWORD *)(*(_QWORD *)v5 + 72LL);
          v17 = 0;
          goto LABEL_9;
        }
      }
      else
      {
        if ( !v15 )
          goto LABEL_23;
        v16 = 0;
      }
      v17 = *(_DWORD *)(v15 + 72);
      v18 = v16 | v17;
LABEL_9:
      if ( !v18 || v17 == v16 )
      {
LABEL_23:
        v21 = v6;
        v20 = sub_3353760(v5, v12, a1[21]);
        v2 = (__int64 *)a1[2];
        LODWORD(v6) = v21;
        if ( v20 )
        {
          v11 = &v2[v9];
          v7 = v8;
          v5 = v2[v8];
        }
        else
        {
          v11 = &v2[v10];
          v5 = v2[v7];
        }
        goto LABEL_21;
      }
      if ( v16 && (v17 > v16 || !v17) )
      {
        v5 = v2[v8];
        v7 = v8;
        goto LABEL_21;
      }
      ++v8;
      v11 = &v2[v10];
      if ( v8 == (_DWORD)v6 )
        goto LABEL_15;
    }
    if ( v13 < v14 )
    {
      v5 = v2[v8];
      v7 = v8;
    }
    else
    {
      v11 = &v2[v10];
    }
LABEL_21:
    ++v8;
  }
  while ( v8 != (_DWORD)v6 );
LABEL_15:
  v1 = (__int64 *)a1[3];
  if ( v7 + 1 != v1 - v2 )
  {
    *v11 = *(v1 - 1);
    *(v1 - 1) = v5;
    v1 = (__int64 *)a1[3];
  }
LABEL_17:
  a1[3] = v1 - 1;
  *(_DWORD *)(v5 + 204) = 0;
  return v5;
}
