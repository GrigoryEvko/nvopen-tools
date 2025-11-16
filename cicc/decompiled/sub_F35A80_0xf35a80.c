// Function: sub_F35A80
// Address: 0xf35a80
//
_BYTE *__fastcall sub_F35A80(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  _BYTE *v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r8
  int v13; // ecx
  int v14; // eax
  __int64 v15; // r13
  _BYTE *v16; // rax
  _BYTE *v17; // r8
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE *v25; // [rsp-40h] [rbp-40h]
  _BYTE *v26; // [rsp-40h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 56);
  if ( !v3 )
    BUG();
  if ( *(_BYTE *)(v3 - 24) != 84 )
  {
    v19 = *(_QWORD *)(a1 + 16);
    while ( v19 )
    {
      v20 = *(_QWORD *)(v19 + 24);
      v19 = *(_QWORD *)(v19 + 8);
      if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
      {
        while ( v19 )
        {
          v21 = *(_QWORD *)(v19 + 24);
          v19 = *(_QWORD *)(v19 + 8);
          if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
          {
            if ( !v19 )
            {
LABEL_24:
              v8 = *(_QWORD *)(v20 + 40);
              v9 = *(_QWORD *)(v21 + 40);
              goto LABEL_5;
            }
            while ( (unsigned __int8)(**(_BYTE **)(v19 + 24) - 30) > 0xAu )
            {
              v19 = *(_QWORD *)(v19 + 8);
              if ( !v19 )
                goto LABEL_24;
            }
            return 0;
          }
        }
        return 0;
      }
    }
    return 0;
  }
  if ( (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) != 2 )
    return 0;
  v6 = *(_QWORD *)(v3 - 32);
  v7 = 32LL * *(unsigned int *)(v3 + 48);
  v8 = *(_QWORD *)(v6 + v7);
  v9 = *(_QWORD *)(v6 + v7 + 8);
LABEL_5:
  v10 = (_BYTE *)sub_986580(v8);
  if ( *v10 != 31 )
    return 0;
  v25 = v10;
  v11 = sub_986580(v9);
  v12 = v11;
  if ( *(_BYTE *)v11 != 31 )
    return 0;
  v13 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
  v14 = *((_DWORD *)v25 + 1) & 0x7FFFFFF;
  if ( v13 == 3 )
  {
    if ( v14 == 3 )
      return 0;
    goto LABEL_26;
  }
  if ( v14 == 3 )
  {
    v24 = v9;
    v12 = (unsigned __int64)v25;
    v9 = v8;
    v8 = v24;
LABEL_26:
    v26 = (_BYTE *)v12;
    if ( sub_AA54C0(v8) )
    {
      v17 = v26;
      v22 = *((_QWORD *)v26 - 4);
      if ( v22 && a1 == v22 && v8 == *((_QWORD *)v26 - 8) )
        goto LABEL_13;
      if ( v8 == v22 )
      {
        v23 = *((_QWORD *)v26 - 8);
        if ( v23 )
        {
          if ( a1 == v23 )
          {
LABEL_32:
            *a2 = v8;
            *a3 = v9;
            return v17;
          }
        }
      }
    }
    return 0;
  }
  v15 = sub_AA54C0(v8);
  if ( !v15 )
    return 0;
  if ( v15 != sub_AA54C0(v9) )
    return 0;
  v16 = (_BYTE *)sub_986580(v15);
  v17 = v16;
  if ( *v16 != 31 )
    return 0;
  if ( *((_QWORD *)v16 - 4) == v8 )
    goto LABEL_32;
LABEL_13:
  *a2 = v9;
  *a3 = v8;
  return v17;
}
