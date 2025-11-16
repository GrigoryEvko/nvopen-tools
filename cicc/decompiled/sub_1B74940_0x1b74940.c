// Function: sub_1B74940
// Address: 0x1b74940
//
__int64 __fastcall sub_1B74940(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v8; // rsi
  __int64 v9; // r8
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  _QWORD *v12; // r10
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned int v15; // edi
  _QWORD *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdi
  _QWORD *v19; // rax
  int v20; // ecx
  int v21; // r11d
  int v22; // eax
  int v23; // r11d
  _QWORD *v24; // [rsp+8h] [rbp-38h]
  unsigned __int64 v25[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v26; // [rsp+20h] [rbp-20h]

  if ( !a3 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    return a1;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 24LL) + 16LL * *(unsigned int *)(*(_QWORD *)a2 + 16LL));
  if ( !*(_BYTE *)(v4 + 64) )
    goto LABEL_3;
  v8 = *(unsigned int *)(v4 + 56);
  if ( !(_DWORD)v8 )
    goto LABEL_3;
  v9 = *(_QWORD *)(v4 + 40);
  v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v11 = (_QWORD *)(v9 + 16LL * v10);
  v12 = (_QWORD *)*v11;
  if ( a3 != (_QWORD *)*v11 )
  {
    v20 = 1;
    while ( v12 != (_QWORD *)-4LL )
    {
      v21 = v20 + 1;
      v10 = (v8 - 1) & (v20 + v10);
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = (_QWORD *)*v11;
      if ( a3 == (_QWORD *)*v11 )
        goto LABEL_13;
      v20 = v21;
    }
    goto LABEL_3;
  }
LABEL_13:
  if ( v11 == (_QWORD *)(v9 + 16 * v8) )
  {
LABEL_3:
    if ( !*(_BYTE *)a3 )
      goto LABEL_8;
    if ( *(_BYTE *)a3 == 1 )
    {
      v5 = *(unsigned int *)(v4 + 24);
      v6 = a3[17];
      if ( !(_DWORD)v5 )
        goto LABEL_6;
      v14 = *(_QWORD *)(v4 + 8);
      v15 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v16 = (_QWORD *)(v14 + ((unsigned __int64)v15 << 6));
      v17 = v16[3];
      if ( v6 != v17 )
      {
        v22 = 1;
        while ( v17 != -8 )
        {
          v23 = v22 + 1;
          v15 = (v5 - 1) & (v22 + v15);
          v16 = (_QWORD *)(v14 + ((unsigned __int64)v15 << 6));
          v17 = v16[3];
          if ( v6 == v17 )
            goto LABEL_17;
          v22 = v23;
        }
        goto LABEL_6;
      }
LABEL_17:
      if ( v16 == (_QWORD *)(v14 + (v5 << 6)) )
      {
LABEL_6:
        if ( v6 )
          a3 = 0;
        goto LABEL_8;
      }
      v18 = v16[7];
      v25[0] = 6;
      v25[1] = 0;
      v26 = v18;
      if ( v18 != -8 && v18 != 0 && v18 != -16 )
      {
        v24 = a3;
        sub_1649AC0(v25, v16[5] & 0xFFFFFFFFFFFFFFF8LL);
        a3 = v24;
        v18 = v26;
        v6 = v24[17];
      }
      if ( v18 == v6 )
        goto LABEL_24;
      if ( v18 )
      {
        v19 = sub_1624210(v18);
        v6 = v26;
        a3 = v19;
LABEL_24:
        *(_QWORD *)a1 = a3;
        *(_BYTE *)(a1 + 8) = 1;
        if ( v6 != 0 && v6 != -8 && v6 != -16 )
          sub_1649B30(v25);
        return a1;
      }
      a3 = 0;
LABEL_8:
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = a3;
      return a1;
    }
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
  v13 = v11[1];
  *(_BYTE *)(a1 + 8) = 1;
  *(_QWORD *)a1 = v13;
  return a1;
}
