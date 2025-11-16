// Function: sub_1D66F70
// Address: 0x1d66f70
//
bool __fastcall sub_1D66F70(_QWORD **a1, __int64 a2)
{
  char v3; // al
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // eax
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // eax
  int v18; // eax
  _QWORD *v19; // rax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdx

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 != 51 )
  {
    if ( v3 != 5 || *(_WORD *)(a2 + 18) != 27 )
      return 0;
    v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v13 = *(_QWORD *)(a2 - 24 * v12);
    v14 = *(_QWORD *)(v13 + 8);
    if ( !v14 || *(_QWORD *)(v14 + 8) )
      goto LABEL_17;
    v23 = *(unsigned __int8 *)(v13 + 16);
    if ( (unsigned __int8)v23 > 0x17u )
    {
      v24 = v23 - 24;
    }
    else
    {
      if ( (_BYTE)v23 != 5 )
        goto LABEL_17;
      v24 = *(unsigned __int16 *)(v13 + 18);
    }
    if ( v24 != 37 )
      goto LABEL_17;
    v25 = *(_QWORD *)sub_13CF970(v13);
    if ( !v25 )
      goto LABEL_17;
    **a1 = v25;
    if ( !sub_1D66D60((__int64)(a1 + 1), *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    {
      v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_17:
      v15 = *(_QWORD *)(a2 + 24 * (1 - v12));
      v16 = *(_QWORD *)(v15 + 8);
      if ( !v16 || *(_QWORD *)(v16 + 8) )
        return 0;
      v17 = *(unsigned __int8 *)(v15 + 16);
      if ( (unsigned __int8)v17 > 0x17u )
      {
        v18 = v17 - 24;
      }
      else
      {
        if ( (_BYTE)v17 != 5 )
          return 0;
        v18 = *(unsigned __int16 *)(v15 + 18);
      }
      if ( v18 == 37 )
      {
        v19 = (_QWORD *)sub_13CF970(v15);
        if ( *v19 )
        {
          **a1 = *v19;
          return sub_1D66D60((__int64)(a1 + 1), *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        }
      }
      return 0;
    }
    return 1;
  }
  v5 = *(_QWORD *)(a2 - 48);
  v6 = *(_QWORD *)(v5 + 8);
  if ( v6 && !*(_QWORD *)(v6 + 8) )
  {
    v20 = *(unsigned __int8 *)(v5 + 16);
    if ( (unsigned __int8)v20 > 0x17u )
    {
      v21 = v20 - 24;
    }
    else
    {
      if ( (_BYTE)v20 != 5 )
        goto LABEL_7;
      v21 = *(unsigned __int16 *)(v5 + 18);
    }
    if ( v21 == 37 )
    {
      v22 = *(_QWORD *)sub_13CF970(v5);
      if ( v22 )
      {
        **a1 = v22;
        if ( sub_1D66B50((__int64)(a1 + 1), *(_QWORD *)(a2 - 24)) )
          return 1;
      }
    }
  }
LABEL_7:
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(_QWORD *)(v7 + 8);
  if ( !v8 || *(_QWORD *)(v8 + 8) )
    return 0;
  v9 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v9 > 0x17u )
  {
    v10 = v9 - 24;
  }
  else
  {
    if ( (_BYTE)v9 != 5 )
      return 0;
    v10 = *(unsigned __int16 *)(v7 + 18);
  }
  if ( v10 != 37 )
    return 0;
  v11 = *(_QWORD *)sub_13CF970(v7);
  if ( !v11 )
    return 0;
  **a1 = v11;
  return sub_1D66B50((__int64)(a1 + 1), *(_QWORD *)(a2 - 48));
}
