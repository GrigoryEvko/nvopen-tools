// Function: sub_10C9A60
// Address: 0x10c9a60
//
bool __fastcall sub_10C9A60(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // r13
  __int64 v5; // r15
  unsigned int v6; // r14d
  int v7; // eax
  bool v8; // al
  __int64 v9; // r13
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  unsigned int v14; // r14d
  int v15; // eax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  int v18; // r14d
  unsigned int v19; // ebx
  __int64 v20; // rax
  int v21; // eax
  unsigned __int8 *v22; // [rsp-50h] [rbp-50h]
  bool v23; // [rsp-41h] [rbp-41h]
  unsigned __int8 *v24; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v25; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v26; // [rsp-40h] [rbp-40h]
  int v27; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 44 )
    return 0;
  v5 = *((_QWORD *)v4 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( v6 <= 0x40 )
    {
      v8 = *(_QWORD *)(v5 + 24) == 0;
    }
    else
    {
      v24 = a3;
      v7 = sub_C444A0(v5 + 24);
      a3 = v24;
      v8 = v6 == v7;
    }
  }
  else
  {
    v12 = *(_QWORD *)(v5 + 8);
    v25 = a3;
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v13 = sub_AD7630(v5, 0, (__int64)a3);
    a3 = v25;
    if ( !v13 || *v13 != 17 )
    {
      if ( *(_BYTE *)(v12 + 8) == 17 )
      {
        v18 = *(_DWORD *)(v12 + 32);
        if ( v18 )
        {
          v23 = 0;
          v19 = 0;
          while ( 1 )
          {
            v26 = a3;
            v20 = sub_AD69F0((unsigned __int8 *)v5, v19);
            if ( !v20 )
              break;
            a3 = v26;
            if ( *(_BYTE *)v20 != 13 )
            {
              if ( *(_BYTE *)v20 != 17 )
                break;
              if ( *(_DWORD *)(v20 + 32) <= 0x40u )
              {
                v23 = *(_QWORD *)(v20 + 24) == 0;
              }
              else
              {
                v22 = v26;
                v27 = *(_DWORD *)(v20 + 32);
                v21 = sub_C444A0(v20 + 24);
                a3 = v22;
                v23 = v27 == v21;
              }
              if ( !v23 )
                break;
            }
            if ( v18 == ++v19 )
            {
              if ( v23 )
                goto LABEL_9;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v14 = *((_DWORD *)v13 + 8);
    if ( v14 <= 0x40 )
    {
      v8 = *((_QWORD *)v13 + 3) == 0;
    }
    else
    {
      v15 = sub_C444A0((__int64)(v13 + 24));
      a3 = v25;
      v8 = v14 == v15;
    }
  }
  if ( !v8 )
    return 0;
LABEL_9:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  if ( *((_QWORD *)v4 - 4) != *(_QWORD *)(a1 + 8) )
    return 0;
  v9 = *((_QWORD *)a3 - 4);
  if ( !v9 )
    BUG();
  if ( *(_BYTE *)v9 != 17 )
  {
    v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
    if ( (unsigned int)v16 > 1 )
      return 0;
    if ( *(_BYTE *)v9 > 0x15u )
      return 0;
    v17 = sub_AD7630(v9, 0, v16);
    v9 = (__int64)v17;
    if ( !v17 || *v17 != 17 )
      return 0;
  }
  v10 = *(_DWORD *)(v9 + 32);
  if ( v10 > 0x40 )
  {
    if ( v10 - (unsigned int)sub_C444A0(v9 + 24) <= 0x40 )
    {
      v11 = **(_QWORD **)(v9 + 24);
      return *(_QWORD *)(a1 + 16) == v11;
    }
    return 0;
  }
  v11 = *(_QWORD *)(v9 + 24);
  return *(_QWORD *)(a1 + 16) == v11;
}
