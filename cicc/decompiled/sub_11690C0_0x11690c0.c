// Function: sub_11690C0
// Address: 0x11690c0
//
__int64 __fastcall sub_11690C0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  char *v4; // r12
  __int64 v5; // rax
  char v6; // al
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // r12
  unsigned int v11; // r13d
  bool v12; // al
  __int64 *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdx
  _BYTE *v16; // rax
  unsigned int v17; // r13d
  int v18; // r13d
  char v19; // r14
  unsigned int v20; // r15d
  __int64 v21; // rax
  unsigned int v22; // r14d
  _BYTE *v23; // rax
  __int64 v24; // rcx
  _BYTE *v25; // rdi
  char v26; // al
  unsigned __int8 *v27; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v28; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (char *)*((_QWORD *)a3 - 8);
  v5 = *((_QWORD *)v4 + 2);
  if ( !v5 || *(_QWORD *)(v5 + 8) )
    return 0;
  v6 = *v4;
  if ( *v4 == 67 )
  {
    v23 = (_BYTE *)*((_QWORD *)v4 - 4);
    if ( *v23 != 55 )
      return 0;
    v24 = *((_QWORD *)v23 - 8);
    if ( !v24 )
      return 0;
    **a1 = v24;
    v25 = (_BYTE *)*((_QWORD *)v23 - 4);
    if ( *v25 <= 0x15u )
    {
      *a1[1] = v25;
      if ( *v25 > 0x15u )
        goto LABEL_14;
      if ( *v25 != 5 )
      {
        v28 = a3;
        v26 = sub_AD6CA0((__int64)v25);
        a3 = v28;
        if ( !v26 )
          goto LABEL_14;
      }
    }
    v6 = *v4;
  }
  if ( v6 != 55 )
    return 0;
  v8 = *((_QWORD *)v4 - 8);
  if ( !v8 )
    return 0;
  *a1[3] = v8;
  v9 = (_BYTE *)*((_QWORD *)v4 - 4);
  if ( *v9 > 0x15u )
    return 0;
  *a1[4] = v9;
  if ( *v9 <= 0x15u )
  {
    v27 = a3;
    if ( *v9 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v9) )
      return 0;
    a3 = v27;
  }
LABEL_14:
  v10 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
      v12 = *(_QWORD *)(v10 + 24) == 1;
    else
      v12 = v11 - 1 == (unsigned int)sub_C444A0(v10 + 24);
  }
  else
  {
    v14 = *(_QWORD *)(v10 + 8);
    v15 = (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17;
    if ( (unsigned int)v15 > 1 || *(_BYTE *)v10 > 0x15u )
      return 0;
    v16 = sub_AD7630(v10, 0, v15);
    if ( !v16 || *v16 != 17 )
    {
      if ( *(_BYTE *)(v14 + 8) == 17 )
      {
        v18 = *(_DWORD *)(v14 + 32);
        if ( v18 )
        {
          v19 = 0;
          v20 = 0;
          while ( 1 )
          {
            v21 = sub_AD69F0((unsigned __int8 *)v10, v20);
            if ( !v21 )
              break;
            if ( *(_BYTE *)v21 != 13 )
            {
              if ( *(_BYTE *)v21 != 17 )
                return 0;
              v22 = *(_DWORD *)(v21 + 32);
              if ( v22 <= 0x40 )
              {
                if ( *(_QWORD *)(v21 + 24) != 1 )
                  return 0;
              }
              else if ( (unsigned int)sub_C444A0(v21 + 24) != v22 - 1 )
              {
                return 0;
              }
              v19 = 1;
            }
            if ( v18 == ++v20 )
            {
              if ( v19 )
                goto LABEL_18;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v17 = *((_DWORD *)v16 + 8);
    if ( v17 <= 0x40 )
    {
      if ( *((_QWORD *)v16 + 3) == 1 )
        goto LABEL_18;
      return 0;
    }
    v12 = v17 - 1 == (unsigned int)sub_C444A0((__int64)(v16 + 24));
  }
  if ( !v12 )
    return 0;
LABEL_18:
  v13 = a1[6];
  if ( v13 )
    *v13 = v10;
  return 1;
}
