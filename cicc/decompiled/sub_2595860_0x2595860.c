// Function: sub_2595860
// Address: 0x2595860
//
__int64 __fastcall sub_2595860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r13
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r13
  char v13; // al
  char v14; // r15
  int v15; // r8d
  __int64 v16; // rsi
  __int64 v17; // rdi
  char v18; // al
  char v19; // al
  bool v20; // zf
  __int64 v21; // rax
  int v23; // [rsp+Ch] [rbp-64h]
  int v24; // [rsp+Ch] [rbp-64h]
  int v25; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v26[2]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v27; // [rsp+20h] [rbp-50h]
  unsigned __int64 v28[2]; // [rsp+28h] [rbp-48h] BYREF
  _BYTE v29[56]; // [rsp+38h] [rbp-38h] BYREF

  v5 = *(_QWORD *)a2;
  v6 = *(unsigned int *)(a2 + 16);
  v28[0] = (unsigned __int64)v29;
  v28[1] = 0;
  v27 = v5;
  if ( (_DWORD)v6 )
  {
    sub_2538240((__int64)v28, (char **)(a2 + 8), a3, a4, v5, v6);
    v5 = v27;
  }
  v7 = *(_QWORD *)a1;
  sub_250D230(v26, v5, 5, 0);
  v8 = v7;
  v9 = 0;
  v10 = sub_25952D0(v8, v26[0], v26[1], *(_QWORD *)(a1 + 8), 0, 0, 1);
  v11 = v10;
  if ( v10 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v13 = *(_BYTE *)(v10 + 136);
    v14 = *(_BYTE *)(v12 + 136);
    v15 = *(_DWORD *)(v12 + 160);
    if ( v13 )
    {
      v13 = *(_BYTE *)(v12 + 136);
    }
    else
    {
      v16 = v11 + 144;
      v17 = v12 + 144;
      if ( !v14 )
      {
        v25 = *(_DWORD *)(v12 + 160);
        sub_2561130(v17, v16);
        v15 = v25;
LABEL_8:
        v18 = *(_BYTE *)(v11 + 136) & *(_BYTE *)(v12 + 136);
        *(_BYTE *)(v12 + 136) = v18;
        v19 = *(_BYTE *)(v12 + 96) | v18;
        if ( v19 )
        {
LABEL_12:
          *(_BYTE *)(v12 + 136) = v19;
          v20 = *(_DWORD *)(v12 + 160) == v15;
          v9 = 1;
          **(_BYTE **)(a1 + 16) |= !v20 | v14 ^ v19;
          v21 = *(_QWORD *)(a1 + 8);
          if ( !*(_DWORD *)(v21 + 160) )
          {
            v9 = *(unsigned __int8 *)(v21 + 136);
            if ( !(_BYTE)v9 )
            {
              v9 = 1;
              if ( !*(_DWORD *)(v21 + 120) )
                v9 = *(unsigned __int8 *)(v21 + 96);
            }
          }
          goto LABEL_14;
        }
LABEL_9:
        v24 = v15;
        sub_2577D20(v12 + 144, v12 + 104);
        v15 = v24;
        v19 = *(_BYTE *)(v12 + 96) | *(_BYTE *)(v12 + 136);
        goto LABEL_12;
      }
      if ( v16 != v17 )
      {
        v23 = *(_DWORD *)(v12 + 160);
        sub_255D9B0(v17, v16);
        v15 = v23;
        goto LABEL_8;
      }
      *(_BYTE *)(v12 + 136) = 0;
    }
    v19 = *(_BYTE *)(v12 + 96) | v13;
    if ( v19 )
      goto LABEL_12;
    goto LABEL_9;
  }
LABEL_14:
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  return v9;
}
