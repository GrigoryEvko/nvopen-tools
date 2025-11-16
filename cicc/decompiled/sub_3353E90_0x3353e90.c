// Function: sub_3353E90
// Address: 0x3353e90
//
__int64 __fastcall sub_3353E90(_QWORD *a1)
{
  __int64 *v1; // rax
  __int64 *v2; // rsi
  __int64 v3; // r15
  __int64 v4; // rdx
  unsigned int v5; // r10d
  unsigned int v6; // ebx
  bool v7; // al
  __int64 *v8; // rcx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r8
  unsigned __int8 v12; // dl
  unsigned __int8 v13; // al
  char v15; // al
  __int64 v16; // r9
  __int64 v17; // r8
  int v18; // eax
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  unsigned int v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  unsigned int v22; // [rsp+10h] [rbp-40h]
  char v23; // [rsp+18h] [rbp-38h]
  int v24; // [rsp+1Ch] [rbp-34h]

  v1 = (__int64 *)a1[3];
  v2 = (__int64 *)a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = *v2;
  v4 = (char *)v1 - (char *)v2;
  if ( (unsigned __int64)((char *)v1 - (char *)v2) > 0x1F40 )
  {
    v24 = 1000;
  }
  else
  {
    v24 = v4 >> 3;
    if ( v4 == 8 )
      goto LABEL_16;
  }
  v5 = 0;
  v6 = 1;
  do
  {
    while ( 1 )
    {
      v9 = v6;
      v10 = v5;
      v8 = &v2[v9];
      v11 = v2[v9];
      v12 = (*(_BYTE *)(v3 + 249) & 0x10) != 0;
      v13 = (*(_BYTE *)(v11 + 249) & 0x10) != 0;
      if ( v12 != v13 )
        break;
      if ( (*(_BYTE *)(v3 + 248) & 2) != 0 || (*(_BYTE *)(v11 + 248) & 2) != 0 )
      {
        v20 = v5;
        v7 = sub_3353760(v3, v2[v6], a1[21]);
        v5 = v20;
      }
      else
      {
        v19 = v5;
        v21 = v2[v6];
        v23 = sub_3351CF0(a1[21], v3);
        v15 = sub_3351CF0(a1[21], v21);
        v17 = v21;
        v5 = v19;
        if ( v15 != 1 && v23 )
        {
          v2 = (__int64 *)a1[2];
          v5 = v6;
          v8 = &v2[v9];
          v3 = v2[v6];
          goto LABEL_10;
        }
        if ( v23 != 1 )
        {
          if ( v15 )
          {
            v2 = (__int64 *)a1[2];
            v8 = &v2[v10];
            v3 = v2[v10];
            goto LABEL_10;
          }
          if ( !v23 )
          {
            v18 = sub_33532E0(v3, v21, 1, a1[21], v21, v16);
            v17 = v21;
            v5 = v19;
            if ( v18 )
            {
              v2 = (__int64 *)a1[2];
              if ( v18 > 0 )
              {
LABEL_26:
                v8 = &v2[v9];
                v5 = v6;
                v3 = v2[v6];
                goto LABEL_10;
              }
              goto LABEL_9;
            }
          }
        }
        v22 = v5;
        v7 = sub_3353760(v3, v17, a1[21]);
        v5 = v22;
      }
      v2 = (__int64 *)a1[2];
      if ( v7 )
        goto LABEL_26;
LABEL_9:
      v8 = &v2[v10];
      v3 = v2[v10];
LABEL_10:
      if ( ++v6 == v24 )
        goto LABEL_14;
    }
    if ( v12 < v13 )
    {
      v3 = v2[v6];
      v5 = v6;
      goto LABEL_10;
    }
    v8 = &v2[v10];
    ++v6;
  }
  while ( v6 != v24 );
LABEL_14:
  v1 = (__int64 *)a1[3];
  if ( v5 + 1 != v1 - v2 )
  {
    *v8 = *(v1 - 1);
    *(v1 - 1) = v3;
    v1 = (__int64 *)a1[3];
  }
LABEL_16:
  a1[3] = v1 - 1;
  *(_DWORD *)(v3 + 204) = 0;
  return v3;
}
