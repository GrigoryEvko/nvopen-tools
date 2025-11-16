// Function: sub_3353B60
// Address: 0x3353b60
//
__int64 __fastcall sub_3353B60(_QWORD *a1)
{
  char *v1; // rax
  char *v2; // rcx
  __int64 v3; // rdx
  unsigned int v5; // r15d
  unsigned int v6; // ebx
  bool v7; // al
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // rdi
  unsigned __int8 v11; // al
  unsigned __int8 v12; // dl
  char *v13; // r14
  __int64 v14; // r8
  int v16; // [rsp-3Ch] [rbp-3Ch]

  v1 = (char *)a1[3];
  v2 = (char *)a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = v1 - v2;
  if ( (unsigned __int64)(v1 - v2) > 0x1F40 )
  {
    v16 = 1000;
  }
  else
  {
    v16 = v3 >> 3;
    if ( v3 == 8 )
    {
      v14 = *(_QWORD *)v2;
      goto LABEL_12;
    }
  }
  v5 = 0;
  v6 = 1;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)&v2[8 * v6];
      v9 = 8LL * v6;
      v10 = *(_QWORD *)&v2[8 * v5];
      v11 = (*(_BYTE *)(v8 + 249) & 0x10) != 0;
      v12 = (*(_BYTE *)(v10 + 249) & 0x10) != 0;
      if ( v12 != v11 )
        break;
      v7 = sub_3353760(v10, v8, a1[21]);
      v2 = (char *)a1[2];
      if ( !v7 )
        goto LABEL_9;
LABEL_6:
      v5 = v6++;
      if ( v16 == v6 )
        goto LABEL_10;
    }
    if ( v12 < v11 )
      goto LABEL_6;
LABEL_9:
    v9 = 8LL * v5;
    ++v6;
  }
  while ( v16 != v6 );
LABEL_10:
  v1 = (char *)a1[3];
  v13 = &v2[v9];
  v14 = *(_QWORD *)v13;
  if ( v5 + 1 != (v1 - v2) >> 3 )
  {
    *(_QWORD *)v13 = *((_QWORD *)v1 - 1);
    *((_QWORD *)v1 - 1) = v14;
    v1 = (char *)a1[3];
  }
LABEL_12:
  a1[3] = v1 - 8;
  *(_DWORD *)(v14 + 204) = 0;
  return v14;
}
