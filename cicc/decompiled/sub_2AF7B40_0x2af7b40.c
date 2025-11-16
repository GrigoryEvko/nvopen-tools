// Function: sub_2AF7B40
// Address: 0x2af7b40
//
__int64 __fastcall sub_2AF7B40(__int64 a1, __int64 a2, unsigned __int64 a3, char a4, int a5)
{
  unsigned int v5; // ebx
  unsigned int v6; // r13d
  char v7; // r15
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 *v10; // r12
  char v11; // al
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 i; // rcx
  unsigned __int64 v16; // rdx
  __int16 v19; // dx
  __int16 v20; // cx
  char v21; // si
  __int64 v22; // r12
  char v23; // al
  __int64 v24; // rax
  int v25; // eax
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  double v30; // xmm0_8
  double v31; // xmm0_8
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+10h] [rbp-70h]
  unsigned __int64 v45; // [rsp+38h] [rbp-48h]
  int v46[13]; // [rsp+4Ch] [rbp-34h] BYREF

  if ( !a3 )
    return 0;
  v5 = 0;
  v6 = 0;
  v7 = 0;
  v45 = 1LL << a4;
  v8 = 0;
  v9 = 0;
  do
  {
    v10 = (__int64 *)(a2 + 8 * v8);
    v11 = sub_2AF69D0(a1, *v10);
    v12 = 1LL << v11;
    if ( 1LL << v11 > (unsigned __int64)(1LL << v7) )
    {
      v7 = v11;
      v9 = v5;
    }
    if ( v45 )
    {
      LODWORD(v13) = v45;
      v14 = (unsigned int)v45;
      if ( v6 )
      {
        v13 = v6;
        for ( i = v45 % v6; i; i = v16 )
        {
          v16 = v13 % i;
          v13 = i;
        }
        v14 = (unsigned int)v13;
      }
    }
    else
    {
      LODWORD(v13) = v6;
      v14 = v6;
    }
    if ( (_DWORD)v13 && v12 < v14 && ((unsigned int)v13 & ((_DWORD)v13 - 1)) == 0 )
    {
      if ( v14 )
      {
        _BitScanReverse64(&v14, v14);
        v19 = v14 ^ 0x3F;
        v20 = 63 - v19;
        v21 = 63 - v19;
      }
      else
      {
        v21 = -1;
        v20 = -1;
      }
      v22 = *v10;
      v23 = *(_BYTE *)v22;
      if ( *(_BYTE *)v22 == 61 || v23 == 62 )
      {
        *(_WORD *)(v22 + 2) = (2 * v20) & 0x1FE | *(_WORD *)(v22 + 2) & 0xFF81;
        goto LABEL_13;
      }
      if ( v23 == 85 )
      {
        v24 = *(_QWORD *)(v22 - 32);
        if ( !v24 || *(_BYTE *)v24 || *(_QWORD *)(v24 + 24) != *(_QWORD *)(v22 + 80) )
          BUG();
        v25 = *(_DWORD *)(v24 + 36);
        if ( v25 == 8975 || v25 == 9567 )
        {
          v28 = *(_QWORD *)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF));
          v29 = *(_QWORD **)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) > 0x40u )
            v29 = (_QWORD *)*v29;
          v41 = (int)v29;
          if ( 1LL << v20 < 0 )
          {
            v36 = (1LL << v20) & 1 | ((unsigned __int64)(1LL << v20) >> 1);
            v30 = (double)(int)v36 + (double)(int)v36;
          }
          else
          {
            v30 = (double)(int)(1LL << v20);
          }
          v31 = log2(v30);
          v32 = sub_AD64C0(
                  *(_QWORD *)(*(_QWORD *)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF)) + 8LL),
                  v41 & 0xFFFC1FFF | (((int)(v31 + 1.0) & 0x1F) << 13),
                  0);
          v33 = v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF);
          if ( *(_QWORD *)v33 )
          {
            v34 = *(_QWORD *)(v33 + 8);
            **(_QWORD **)(v33 + 16) = v34;
            if ( v34 )
              *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
          }
          *(_QWORD *)v33 = v32;
          if ( v32 )
          {
            v35 = *(_QWORD *)(v32 + 16);
            *(_QWORD *)(v33 + 8) = v35;
            if ( v35 )
              *(_QWORD *)(v35 + 16) = v33 + 8;
            *(_QWORD *)(v33 + 16) = v32 + 16;
            *(_QWORD *)(v32 + 16) = v33;
          }
        }
        else
        {
          if ( v25 == 8937 )
          {
            v37 = (__int64 *)sub_BD5C60(v22);
            v38 = sub_A77A40(v37, v21);
            v46[0] = 1;
            v40 = v38;
          }
          else
          {
            if ( v25 != 9549 )
              goto LABEL_13;
            v26 = (__int64 *)sub_BD5C60(v22);
            v27 = sub_A77A40(v26, v21);
            v46[0] = 2;
            v40 = v27;
          }
          v39 = (__int64 *)sub_BD5C60(v22);
          *(_QWORD *)(v22 + 72) = sub_A7B660((__int64 *)(v22 + 72), v39, v46, 1, v40);
        }
      }
    }
LABEL_13:
    v6 += a5;
    v8 = ++v5;
  }
  while ( v5 < a3 );
  return v9;
}
