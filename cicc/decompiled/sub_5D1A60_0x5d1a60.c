// Function: sub_5D1A60
// Address: 0x5d1a60
//
__int64 __fastcall sub_5D1A60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r14
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  _QWORD *v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  _DWORD v30[13]; // [rsp+2Ch] [rbp-34h] BYREF

  if ( unk_4D045E8 > 0x59u )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_QWORD **)v2;
    v4 = *(_QWORD *)(v2 + 40);
    if ( *(_QWORD *)v2 )
      goto LABEL_3;
LABEL_23:
    v27 = 0;
    v6 = 0;
    v28 = 0;
    v29 = 0;
    goto LABEL_8;
  }
  sub_684AA0(7, 3790, a1 + 56);
  v14 = *(_QWORD **)(a1 + 32);
  v3 = (_QWORD *)*v14;
  v4 = v14[5];
  if ( !*v14 )
    goto LABEL_23;
LABEL_3:
  v5 = v3[5];
  v3 = (_QWORD *)*v3;
  v29 = v5;
  if ( v3 )
  {
    v6 = v3[5];
    v3 = (_QWORD *)*v3;
    if ( v3 )
    {
      v7 = v3[5];
      v3 = (_QWORD *)*v3;
      v28 = v7;
      if ( v3 )
      {
        v8 = v3[5];
        v3 = (_QWORD *)*v3;
        v27 = v8;
        if ( v3 )
          v3 = (_QWORD *)v3[5];
      }
      else
      {
        v27 = 0;
      }
    }
    else
    {
      v27 = 0;
      v28 = 0;
    }
  }
  else
  {
    v27 = 0;
    v6 = 0;
    v28 = 0;
  }
LABEL_8:
  if ( (unsigned int)sub_5D19D0(a1) )
    return a2;
  v9 = *(_QWORD *)(a2 + 328);
  if ( !v9 )
  {
    v23 = sub_725F60();
    *(_QWORD *)(a2 + 328) = v23;
    v9 = v23;
  }
  if ( v4 )
  {
    v26 = v9;
    if ( (int)sub_6210B0(v4, 0) <= 0 )
    {
      sub_6849F0(7, 3788, a1 + 56, "__block_size__");
    }
    else
    {
      v10 = sub_620FD0(v4, v30);
      if ( v30[0] || v10 > 0x7FFFFFFF )
        sub_684AA0(7, 3789, a1 + 56);
      else
        *(_DWORD *)(v26 + 40) = v10;
    }
  }
  else
  {
    *(_DWORD *)(v9 + 40) = 1;
  }
  v15 = *(_QWORD *)(a2 + 328);
  if ( v29 )
  {
    if ( (int)sub_6210B0(v29, 0) <= 0 )
    {
      sub_6849F0(7, 3788, a1 + 56, "__block_size__");
    }
    else
    {
      v16 = sub_620FD0(v29, v30);
      if ( v30[0] || v16 > 0x7FFFFFFF )
        sub_684AA0(7, 3789, a1 + 56);
      else
        *(_DWORD *)(v15 + 44) = v16;
    }
  }
  else
  {
    *(_DWORD *)(v15 + 44) = 1;
  }
  v17 = *(_QWORD *)(a2 + 328);
  if ( !v6 )
  {
    *(_DWORD *)(v17 + 48) = 1;
    goto LABEL_17;
  }
  if ( (int)sub_6210B0(v6, 0) > 0 )
  {
    v18 = sub_620FD0(v6, v30);
    if ( v30[0] || v18 > 0x7FFFFFFF )
      sub_684AA0(7, 3789, a1 + 56);
    else
      *(_DWORD *)(v17 + 48) = v18;
LABEL_17:
    v11 = *(_QWORD *)(a2 + 328);
    if ( v28 )
      goto LABEL_18;
LABEL_38:
    *(_BYTE *)(v11 + 52) &= ~2u;
    v19 = *(_QWORD *)(a2 + 328);
    if ( (*(_BYTE *)(v19 + 52) & 1) != 0 )
      return a2;
    *(_DWORD *)(v19 + 20) = 1;
    goto LABEL_40;
  }
  sub_6849F0(7, 3788, a1 + 56, "__block_size__");
  v11 = *(_QWORD *)(a2 + 328);
  if ( !v28 )
    goto LABEL_38;
LABEL_18:
  if ( (*(_BYTE *)(v11 + 52) & 1) != 0 )
  {
    sub_684AA0(7, 3791, a1 + 56);
    v11 = *(_QWORD *)(a2 + 328);
  }
  *(_BYTE *)(v11 + 52) |= 2u;
  v12 = *(_QWORD *)(a2 + 328);
  if ( (*(_BYTE *)(v12 + 52) & 1) == 0 )
  {
    if ( (int)sub_6210B0(v28, 0) <= 0 )
    {
      sub_6849F0(7, 3788, a1 + 56, "__block_size__");
    }
    else
    {
      v22 = sub_620FD0(v28, v30);
      if ( v30[0] || v22 > 0x7FFFFFFF )
        sub_684AA0(7, 3789, a1 + 56);
      else
        *(_DWORD *)(v12 + 20) = v22;
    }
LABEL_40:
    v20 = *(_QWORD *)(a2 + 328);
    if ( v27 )
    {
      if ( (int)sub_6210B0(v27, 0) <= 0 )
      {
        sub_6849F0(7, 3788, a1 + 56, "__block_size__");
      }
      else
      {
        v21 = sub_620FD0(v27, v30);
        if ( v30[0] || v21 > 0x7FFFFFFF )
          sub_684AA0(7, 3789, a1 + 56);
        else
          *(_DWORD *)(v20 + 24) = v21;
      }
    }
    else
    {
      *(_DWORD *)(v20 + 24) = 1;
    }
    v24 = *(_QWORD *)(a2 + 328);
    if ( v3 )
    {
      if ( (int)sub_6210B0(v3, 0) <= 0 )
      {
        sub_6849F0(7, 3788, a1 + 56, "__block_size__");
      }
      else
      {
        v25 = sub_620FD0(v3, v30);
        if ( v30[0] || v25 > 0x7FFFFFFF )
          sub_684AA0(7, 3789, a1 + 56);
        else
          *(_DWORD *)(v24 + 28) = v25;
      }
    }
    else
    {
      *(_DWORD *)(v24 + 28) = 1;
    }
  }
  return a2;
}
