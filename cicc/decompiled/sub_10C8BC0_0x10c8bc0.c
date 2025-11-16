// Function: sub_10C8BC0
// Address: 0x10c8bc0
//
__int64 __fastcall sub_10C8BC0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r14d
  int v10; // eax
  bool v11; // al
  _QWORD *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // r14
  _BYTE *v17; // rax
  unsigned int v18; // r14d
  int v19; // eax
  int v20; // r14d
  char v21; // cl
  unsigned int v22; // r15d
  __int64 v23; // rax
  int v24; // eax
  unsigned __int8 *v25; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v26; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v27; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v28; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v29; // [rsp-40h] [rbp-40h]
  char v30; // [rsp-40h] [rbp-40h]
  int v31; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( !v4 || (**a1 = v4, (v5 = *(_QWORD *)(v4 + 16)) == 0) || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 44 )
  {
LABEL_5:
    v6 = *((_QWORD *)a3 - 4);
    if ( v6 )
    {
      **a1 = v6;
      v7 = *(_QWORD *)(v6 + 16);
      if ( v7 )
      {
        if ( !*(_QWORD *)(v7 + 8) && *(_BYTE *)v6 == 44 )
        {
          v28 = a3;
          if ( (unsigned __int8)sub_10081F0(a1 + 1, *(_QWORD *)(v6 - 64)) )
          {
            v15 = *(_BYTE **)(v6 - 32);
            if ( *v15 == 57 )
            {
              result = sub_993A50(a1 + 3, *((_QWORD *)v15 - 4));
              if ( (_BYTE)result )
              {
                v14 = *((_QWORD *)v28 - 8);
                if ( v14 )
                  goto LABEL_19;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  v8 = *(_QWORD *)(v4 - 64);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      v11 = *(_QWORD *)(v8 + 24) == 0;
    }
    else
    {
      v26 = a3;
      v10 = sub_C444A0(v8 + 24);
      a3 = v26;
      v11 = v9 == v10;
    }
    goto LABEL_13;
  }
  v16 = *(_QWORD *)(v8 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 || *(_BYTE *)v8 > 0x15u )
    goto LABEL_5;
  v29 = a3;
  v17 = sub_AD7630(*(_QWORD *)(v4 - 64), 0, (__int64)a3);
  a3 = v29;
  if ( !v17 || *v17 != 17 )
  {
    if ( *(_BYTE *)(v16 + 8) == 17 )
    {
      v20 = *(_DWORD *)(v16 + 32);
      if ( v20 )
      {
        v21 = 0;
        v22 = 0;
        while ( 1 )
        {
          v25 = a3;
          v30 = v21;
          v23 = sub_AD69F0((unsigned __int8 *)v8, v22);
          a3 = v25;
          if ( !v23 )
            break;
          v21 = v30;
          if ( *(_BYTE *)v23 != 13 )
          {
            if ( *(_BYTE *)v23 != 17 )
              goto LABEL_5;
            if ( *(_DWORD *)(v23 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v23 + 24) )
                goto LABEL_5;
            }
            else
            {
              v31 = *(_DWORD *)(v23 + 32);
              v24 = sub_C444A0(v23 + 24);
              a3 = v25;
              if ( v31 != v24 )
                goto LABEL_5;
            }
            v21 = 1;
          }
          if ( v20 == ++v22 )
          {
            if ( v21 )
              goto LABEL_14;
            goto LABEL_5;
          }
        }
      }
    }
    goto LABEL_5;
  }
  v18 = *((_DWORD *)v17 + 8);
  if ( v18 <= 0x40 )
  {
    if ( *((_QWORD *)v17 + 3) )
      goto LABEL_5;
    goto LABEL_14;
  }
  v19 = sub_C444A0((__int64)(v17 + 24));
  a3 = v29;
  v11 = v18 == v19;
LABEL_13:
  if ( !v11 )
    goto LABEL_5;
LABEL_14:
  v12 = a1[1];
  if ( v12 )
    *v12 = v8;
  v13 = *(_BYTE **)(v4 - 32);
  if ( *v13 != 57 )
    goto LABEL_5;
  v27 = a3;
  result = sub_993A50(a1 + 3, *((_QWORD *)v13 - 4));
  a3 = v27;
  if ( !(_BYTE)result )
    goto LABEL_5;
  v14 = *((_QWORD *)v27 - 4);
  if ( !v14 )
    return 0;
LABEL_19:
  *a1[4] = v14;
  return result;
}
