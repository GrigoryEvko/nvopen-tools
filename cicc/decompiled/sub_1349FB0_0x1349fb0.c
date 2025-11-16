// Function: sub_1349FB0
// Address: 0x1349fb0
//
__int64 __fastcall sub_1349FB0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  char *v4; // rax
  _QWORD *v5; // rdx
  unsigned __int64 v6; // r13
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // r14
  __int64 v17; // r10
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // r13
  __int64 v23; // r13
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r9
  unsigned __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  unsigned __int64 *v31; // r9
  __int64 v32; // r11
  __int64 v33; // rdi
  __int64 result; // rax
  unsigned __int64 v35; // r9
  unsigned __int64 v37; // [rsp+8h] [rbp-98h]
  _QWORD *v39; // [rsp+18h] [rbp-88h]
  unsigned __int64 v40; // [rsp+28h] [rbp-78h]
  _OWORD v41[3]; // [rsp+30h] [rbp-70h] BYREF
  __int128 v42; // [rsp+60h] [rbp-40h]
  char v43; // [rsp+70h] [rbp-30h] BYREF

  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a2 + 80) = 0;
  v39 = a1 + 14;
  v3 = 0;
  memset(v41, 0, sizeof(v41));
  v42 = 0;
  do
  {
    *((_QWORD *)v41 + v3) = ~a1[v3 + 14];
    ++v3;
  }
  while ( v3 != 8 );
  v4 = (char *)v41;
  v5 = a1 + 23;
  do
  {
    *(_QWORD *)v4 &= *v5;
    v4 += 8;
    ++v5;
  }
  while ( v4 != &v43 );
  v6 = 0;
  v7 = a2 + 16;
  *(_OWORD *)(a2 + 16) = 0;
  v8 = *((_QWORD *)&v42 + 1);
  *(_OWORD *)(a2 + 32) = 0;
  v37 = v8;
  *(_OWORD *)(a2 + 48) = 0;
  *(_OWORD *)(a2 + 64) = 0;
  do
  {
    v9 = v6 >> 6;
    if ( *((_QWORD *)v41 + (v6 >> 6)) & -(1LL << (v6 & 0x3F)) )
    {
      __asm { tzcnt   rdx, rdx }
      v14 = (v9 << 6) + (int)_RDX;
      goto LABEL_12;
    }
    v11 = v9 + 1;
    if ( v9 == 7 )
      break;
    while ( !*((_QWORD *)v41 + v11) )
    {
      if ( ++v11 == 8 )
        goto LABEL_29;
    }
    __asm { tzcnt   rdx, rdx }
    v14 = (int)_RDX + (v11 << 6);
    if ( v14 == 512 )
      break;
LABEL_12:
    v15 = v14 & 0x3F;
    v16 = v14 >> 6;
    v17 = 8 * (v14 >> 6);
    v18 = v39[(unsigned __int64)v17 / 8] & -(1LL << (v14 & 0x3F));
    if ( v18 )
    {
      v21 = v14 >> 6;
      v19 = v16 + 1;
    }
    else
    {
      v19 = v16 + 1;
      v20 = v16 + 1;
      if ( v16 == 7 )
      {
LABEL_34:
        v6 = 513;
        v27 = 6;
        if ( v37 )
        {
          v26 = v37;
          v29 = 448;
        }
        else
        {
LABEL_23:
          while ( 1 )
          {
            v26 = *((_QWORD *)v41 + v27);
            if ( v26 )
              break;
            if ( v27-- == 0 )
              goto LABEL_33;
          }
          v29 = v27 << 6;
        }
        goto LABEL_25;
      }
      while ( 1 )
      {
        v18 = a1[v20 + 14];
        v21 = v20;
        if ( v18 )
          break;
        if ( ++v20 == 8 )
          goto LABEL_34;
      }
    }
    if ( !_BitScanForward64(&v18, v18) )
      LODWORD(v18) = -1;
    v23 = (int)v18 + (v21 << 6);
    v24 = v23 - 1;
    v6 = v23 + 1;
    v25 = v24 >> 6;
    v26 = *((_QWORD *)v41 + (v24 >> 6)) & ((2LL << (v24 & 0x3F)) - 1);
    if ( v26 )
    {
      v29 = v25 << 6;
LABEL_25:
      _BitScanReverse64(&v26, v26);
      v30 = (int)v26 + v29;
      goto LABEL_26;
    }
    v27 = v25 - 1;
    if ( v25 )
      goto LABEL_23;
LABEL_33:
    v30 = -1;
LABEL_26:
    v31 = (unsigned __int64 *)(v7 + v17);
    v32 = *(_QWORD *)(v7 + v17);
    v33 = 1 - v14 + v30;
    if ( v33 + v15 > 0x40 )
    {
      *v31 = (0xFFFFFFFFFFFFFFFFLL >> v15 << v15) | v32;
      v35 = v33 + v15 - 64;
      if ( v35 <= 0x40 )
      {
        if ( v33 + v15 == 64 )
          continue;
      }
      else
      {
        v40 = (v33 + v15 - 129) >> 6;
        memset((void *)(v7 + v17 + 8), 255, 8 * v40 + 8);
        v19 = v16 + v40 + 2;
        LOBYTE(v35) = v33 + v15 - 64 - ((_BYTE)v40 << 6) - 64;
      }
      *(_QWORD *)(v7 + 8 * v19) |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35);
    }
    else
    {
      *v31 = (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) << v15) | v32;
    }
  }
  while ( v6 <= 0x1FF );
LABEL_29:
  result = a1[22] - a1[13];
  *(_QWORD *)(a2 + 8) = result;
  return result;
}
