// Function: sub_2AC1B80
// Address: 0x2ac1b80
//
__int64 __fastcall sub_2AC1B80(__int64 a1, char a2, _QWORD *a3, _QWORD *a4, unsigned __int8 *a5, __int64 a6)
{
  __int64 v8; // r10
  __int64 v9; // rax
  _QWORD *v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int16 v16; // ax
  __int64 result; // rax
  char v18; // al
  char v19; // dl
  __int64 v20; // rdx
  unsigned __int8 v21; // al
  unsigned __int8 v22; // al
  int v23; // eax
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  _QWORD *v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  _QWORD *v30; // [rsp+20h] [rbp-50h]
  __int64 v31; // [rsp+30h] [rbp-40h] BYREF
  __int64 v32[7]; // [rsp+38h] [rbp-38h] BYREF

  v31 = *((_QWORD *)a5 + 6);
  if ( v31 )
  {
    v30 = a3;
    sub_2AAAFA0(&v31);
    a3 = v30;
    v32[0] = v31;
    if ( v31 )
    {
      sub_2AAAFA0(v32);
      a3 = v30;
    }
  }
  else
  {
    v32[0] = 0;
  }
  *(_BYTE *)(a1 + 8) = a2;
  v8 = a1 + 40;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( a3 != a4 )
  {
    v9 = *a3;
    v10 = a3 + 1;
    v11 = a1 + 64;
    v12 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v11 + 8 * v12) = v9;
      ++*(_DWORD *)(a1 + 56);
      v13 = *(unsigned int *)(v9 + 24);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 28) )
      {
        v25 = v10;
        v27 = v8;
        v29 = v9;
        sub_C8D5F0(v9 + 16, (const void *)(v9 + 32), v13 + 1, 8u, (__int64)v10, a6);
        v9 = v29;
        v10 = v25;
        v8 = v27;
        v13 = *(unsigned int *)(v29 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v13) = v8;
      ++*(_DWORD *)(v9 + 24);
      if ( a4 == v10 )
        break;
      v12 = *(unsigned int *)(a1 + 56);
      v9 = *v10;
      a6 = v12 + 1;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v24 = v10;
        v26 = v8;
        v28 = *v10;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v12 + 1, 8u, (__int64)v10, a6);
        v12 = *(unsigned int *)(a1 + 56);
        v10 = v24;
        v8 = v26;
        v9 = v28;
      }
      v11 = *(_QWORD *)(a1 + 48);
      ++v10;
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v14 = v32[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v14;
  if ( v14 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(v32);
  sub_2BF0340(a1 + 96, 1, a5, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v31);
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  v15 = *a5;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  if ( (unsigned __int8)(v15 - 82) <= 1u )
  {
    v16 = *((_WORD *)a5 + 1);
    *(_BYTE *)(a1 + 152) = 0;
    result = v16 & 0x3F;
    *(_DWORD *)(a1 + 156) = result;
    return result;
  }
  if ( (_BYTE)v15 == 58 )
  {
    *(_BYTE *)(a1 + 152) = 2;
LABEL_20:
    v18 = a5[1] >> 1;
LABEL_21:
    v19 = v18 & 1;
    result = v18 & 1 | *(_BYTE *)(a1 + 156) & 0xFEu;
    *(_BYTE *)(a1 + 156) = v19 | *(_BYTE *)(a1 + 156) & 0xFE;
    return result;
  }
  if ( (unsigned __int8)v15 <= 0x36u )
  {
    v20 = 0x40540000000000LL;
    if ( _bittest64(&v20, v15) )
    {
      v21 = a5[1];
      *(_BYTE *)(a1 + 152) = 1;
      result = *(_BYTE *)(a1 + 156) & 0xFC | (v21 >> 1) & 3u;
      *(_BYTE *)(a1 + 156) = result;
      return result;
    }
    if ( (unsigned int)(unsigned __int8)v15 - 48 <= 1 )
      goto LABEL_24;
  }
  else
  {
    if ( (unsigned __int8)(v15 - 55) <= 1u )
    {
LABEL_24:
      *(_BYTE *)(a1 + 152) = 3;
      goto LABEL_20;
    }
    if ( (_BYTE)v15 == 63 )
    {
      *(_BYTE *)(a1 + 152) = 4;
      result = sub_B4DE20((__int64)a5);
      *(_DWORD *)(a1 + 156) = result;
      return result;
    }
  }
  if ( (((_BYTE)v15 - 68) & 0xFB) == 0 )
  {
    *(_BYTE *)(a1 + 152) = 6;
    v18 = sub_B44910((__int64)a5);
    goto LABEL_21;
  }
  result = sub_920620((__int64)a5);
  if ( (_BYTE)result )
  {
    v22 = a5[1];
    *(_BYTE *)(a1 + 152) = 5;
    v23 = v22 >> 1;
    if ( v23 == 127 )
      v23 = -1;
    LODWORD(v32[0]) = v23;
    sub_2C1AC80(&v31, v32);
    result = (unsigned __int8)v31;
    *(_BYTE *)(a1 + 156) = v31;
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 7;
    *(_DWORD *)(a1 + 156) = 0;
  }
  return result;
}
