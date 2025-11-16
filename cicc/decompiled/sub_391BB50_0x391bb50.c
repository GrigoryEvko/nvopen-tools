// Function: sub_391BB50
// Address: 0x391bb50
//
__int64 __fastcall sub_391BB50(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // rbx
  char v5; // si
  char v6; // al
  char *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdi
  char v11; // si
  char *v12; // rax
  __int64 v13; // rdi
  char v14; // si
  char *v15; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rbx
  char v20; // r14
  char v21; // si
  char *v22; // rax
  char v23; // al
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // [rsp-60h] [rbp-60h]
  _QWORD v27[11]; // [rsp-58h] [rbp-58h] BYREF

  result = *(unsigned int *)(a1 + 624);
  if ( (_DWORD)result )
  {
    sub_391B370(a1, (__int64)v27, 6);
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(unsigned int *)(a1 + 624);
    do
    {
      while ( 1 )
      {
        v5 = v4 & 0x7F;
        v6 = v4 & 0x7F | 0x80;
        v4 >>= 7;
        if ( v4 )
          v5 = v6;
        v7 = *(char **)(v3 + 24);
        if ( (unsigned __int64)v7 >= *(_QWORD *)(v3 + 16) )
          break;
        *(_QWORD *)(v3 + 24) = v7 + 1;
        *v7 = v5;
        if ( !v4 )
          goto LABEL_8;
      }
      sub_16E7DE0(v3, v5);
    }
    while ( v4 );
LABEL_8:
    v8 = *(_QWORD *)(a1 + 616);
    v9 = 16LL * *(unsigned int *)(a1 + 624);
    v26 = v8 + v9;
    if ( v8 + v9 == v8 )
      return sub_3919EA0(a1, v27);
LABEL_9:
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *(_BYTE *)v8;
    v12 = *(char **)(v10 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v10 + 16) )
    {
      sub_16E7DE0(v10, v11);
    }
    else
    {
      *(_QWORD *)(v10 + 24) = v12 + 1;
      *v12 = v11;
    }
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_BYTE *)(v8 + 1);
    v15 = *(char **)(v13 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v13 + 16) )
    {
      sub_16E7DE0(v13, v14);
    }
    else
    {
      *(_QWORD *)(v13 + 24) = v15 + 1;
      *v15 = v14;
    }
    v16 = *(_QWORD *)(a1 + 8);
    v17 = *(_BYTE **)(v16 + 24);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
    {
      sub_16E7DE0(v16, 65);
    }
    else
    {
      *(_QWORD *)(v16 + 24) = v17 + 1;
      *v17 = 65;
    }
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(_QWORD *)(v8 + 8);
    while ( 1 )
    {
      v23 = v19;
      v21 = v19 & 0x7F;
      v19 >>= 7;
      if ( !v19 )
        break;
      if ( v19 == -1 && (v20 = 0, (v23 & 0x40) != 0) )
      {
        v22 = *(char **)(v18 + 24);
        if ( (unsigned __int64)v22 >= *(_QWORD *)(v18 + 16) )
          goto LABEL_24;
LABEL_19:
        *(_QWORD *)(v18 + 24) = v22 + 1;
        *v22 = v21;
        if ( !v20 )
          goto LABEL_25;
      }
      else
      {
LABEL_17:
        v21 |= 0x80u;
        v20 = 1;
LABEL_18:
        v22 = *(char **)(v18 + 24);
        if ( (unsigned __int64)v22 < *(_QWORD *)(v18 + 16) )
          goto LABEL_19;
LABEL_24:
        sub_16E7DE0(v18, v21);
        if ( !v20 )
        {
LABEL_25:
          v24 = *(_QWORD *)(a1 + 8);
          v25 = *(_BYTE **)(v24 + 24);
          if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 16) )
          {
            sub_16E7DE0(v24, 11);
          }
          else
          {
            *(_QWORD *)(v24 + 24) = v25 + 1;
            *v25 = 11;
          }
          v8 += 16;
          if ( v26 == v8 )
            return sub_3919EA0(a1, v27);
          goto LABEL_9;
        }
      }
    }
    v20 = 0;
    if ( (v23 & 0x40) == 0 )
      goto LABEL_18;
    goto LABEL_17;
  }
  return result;
}
