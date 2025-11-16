// Function: sub_2914380
// Address: 0x2914380
//
__int64 __fastcall sub_2914380(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  _QWORD **v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a1 & 4) != 0 )
  {
    v5 = *(_QWORD ***)v3;
    v6 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
    goto LABEL_6;
  }
  if ( v3 )
  {
    v5 = (_QWORD **)a1;
    v6 = (__int64)(a1 + 1);
    while ( 1 )
    {
LABEL_6:
      if ( v5 == (_QWORD **)v6 )
        return a2;
      v7 = *v5;
      if ( *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL) == sub_B12000((__int64)(*v5 + 9)) )
        break;
LABEL_5:
      ++v5;
    }
    v8 = v7[3];
    v17[0] = v8;
    if ( v8 )
      sub_B96E90((__int64)v17, v8, 1);
    v9 = sub_B10CD0((__int64)v17);
    v10 = *(_BYTE *)(v9 - 16);
    if ( (v10 & 2) != 0 )
    {
      if ( *(_DWORD *)(v9 - 24) != 2 )
      {
LABEL_12:
        v16 = 0;
        goto LABEL_13;
      }
      v15 = *(_QWORD *)(v9 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v9 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_12;
      v15 = v9 - 16 - 8LL * ((v10 >> 2) & 0xF);
    }
    v16 = *(_QWORD *)(v15 + 8);
LABEL_13:
    v11 = sub_B10CD0(a2 + 48);
    v12 = *(_BYTE *)(v11 - 16);
    if ( (v12 & 2) != 0 )
    {
      if ( *(_DWORD *)(v11 - 24) != 2 )
        goto LABEL_15;
      v14 = *(_QWORD *)(v11 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v11 - 16) >> 6) & 0xF) != 2 )
      {
LABEL_15:
        v13 = v17[0];
        if ( v16 )
          goto LABEL_16;
LABEL_20:
        if ( v13 )
          sub_B91220((__int64)v17, v13);
        sub_B14290(v7);
        goto LABEL_5;
      }
      v14 = v11 - 16 - 8LL * ((v12 >> 2) & 0xF);
    }
    v13 = v17[0];
    if ( *(_QWORD *)(v14 + 8) != v16 )
    {
LABEL_16:
      if ( v13 )
        sub_B91220((__int64)v17, v13);
      goto LABEL_5;
    }
    goto LABEL_20;
  }
  return a2;
}
