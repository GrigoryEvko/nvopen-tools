// Function: sub_3264EF0
// Address: 0x3264ef0
//
__int64 __fastcall sub_3264EF0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v8; // r13
  int v9; // edx
  unsigned int v10; // r12d
  __int64 v12; // r13
  _DWORD *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // r13
  bool v18; // al
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int16 v29; // cx
  __int16 v32; // [rsp+20h] [rbp-40h] BYREF
  __int64 v33; // [rsp+28h] [rbp-38h]

  v8 = a3;
  v9 = *(_DWORD *)(a2 + 24);
  v10 = a7;
  if ( v9 == 208 )
  {
    v23 = *(_QWORD *)(a2 + 40);
    v10 = 1;
    *(_QWORD *)a4 = *(_QWORD *)v23;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v23 + 8);
    v24 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a5 = *(_QWORD *)(v24 + 40);
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(v24 + 48);
    v25 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a6 = *(_QWORD *)(v25 + 80);
    *(_DWORD *)(a6 + 8) = *(_DWORD *)(v25 + 88);
    return v10;
  }
  if ( (_BYTE)a7 && (unsigned int)(v9 - 147) <= 1 )
  {
    v26 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a4 = *(_QWORD *)(v26 + 40);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v26 + 48);
    v27 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a5 = *(_QWORD *)(v27 + 80);
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(v27 + 88);
    v28 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a6 = *(_QWORD *)(v28 + 120);
    *(_DWORD *)(a6 + 8) = *(_DWORD *)(v28 + 128);
    return v10;
  }
  if ( v9 != 207 )
    return 0;
  if ( !(unsigned __int8)sub_3449B70(
                           *(_QWORD *)(a1 + 8),
                           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL)) )
    return 0;
  v10 = sub_3449EC0(
          *(_QWORD *)(a1 + 8),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 128LL));
  if ( !(_BYTE)v10 )
    return 0;
  v12 = *(_QWORD *)(a2 + 48) + 16 * v8;
  v13 = *(_DWORD **)(a1 + 8);
  v14 = *(_QWORD *)(v12 + 8);
  v15 = a5;
  v16 = a6;
  v32 = *(_WORD *)v12;
  v33 = v14;
  if ( !v32 )
  {
    v17 = sub_3007030((__int64)&v32);
    v18 = sub_30070B0((__int64)&v32);
    v15 = a5;
    v16 = a6;
    if ( !v18 )
    {
      if ( !v17 )
      {
LABEL_12:
        v19 = v13[15];
        goto LABEL_13;
      }
      goto LABEL_22;
    }
    goto LABEL_20;
  }
  v29 = v32 - 17;
  if ( (unsigned __int16)(v32 - 10) > 6u && (unsigned __int16)(v32 - 126) > 0x31u )
  {
    if ( v29 > 0xD3u )
      goto LABEL_12;
    goto LABEL_20;
  }
  if ( v29 <= 0xD3u )
  {
LABEL_20:
    v19 = v13[17];
    goto LABEL_13;
  }
LABEL_22:
  v19 = v13[16];
LABEL_13:
  if ( v19 )
  {
    v20 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a4 = *(_QWORD *)v20;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v20 + 8);
    v21 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)v15 = *(_QWORD *)(v21 + 40);
    *(_DWORD *)(v15 + 8) = *(_DWORD *)(v21 + 48);
    v22 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)v16 = *(_QWORD *)(v22 + 160);
    *(_DWORD *)(v16 + 8) = *(_DWORD *)(v22 + 168);
    return v10;
  }
  return 0;
}
