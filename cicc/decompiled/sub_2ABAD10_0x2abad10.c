// Function: sub_2ABAD10
// Address: 0x2abad10
//
__int64 __fastcall sub_2ABAD10(__int64 a1, char a2, _QWORD *a3, __int64 a4, unsigned __int8 *a5, __int64 a6)
{
  _QWORD *v8; // r11
  __int64 v9; // r8
  __int64 v10; // r13
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int16 v17; // ax
  __int64 result; // rax
  char v19; // al
  char v20; // dl
  __int64 v21; // rdx
  unsigned __int8 v22; // al
  unsigned __int8 v23; // al
  int v24; // eax
  _QWORD *v25; // [rsp+8h] [rbp-68h]
  _QWORD *v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  _QWORD *v30; // [rsp+18h] [rbp-58h]
  _QWORD *v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+38h] [rbp-38h] BYREF

  v32 = *((_QWORD *)a5 + 6);
  if ( v32 )
  {
    v31 = a3;
    sub_2AAAFA0(&v32);
    a3 = v31;
    v33[0] = v32;
    if ( v32 )
    {
      sub_2AAAFA0(v33);
      a3 = v31;
    }
  }
  else
  {
    v33[0] = 0;
  }
  *(_BYTE *)(a1 + 8) = a2;
  v8 = &a3[a4];
  *(_QWORD *)(a1 + 24) = 0;
  v9 = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( a3 != v8 )
  {
    v10 = *a3;
    v11 = a3 + 1;
    v12 = a1 + 64;
    v13 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v12 + 8 * v13) = v10;
      ++*(_DWORD *)(a1 + 56);
      v14 = *(unsigned int *)(v10 + 24);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
      {
        v26 = v8;
        v28 = v9;
        v30 = v11;
        sub_C8D5F0(v10 + 16, (const void *)(v10 + 32), v14 + 1, 8u, v9, a6);
        v14 = *(unsigned int *)(v10 + 24);
        v8 = v26;
        v9 = v28;
        v11 = v30;
      }
      *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v14) = v9;
      ++*(_DWORD *)(v10 + 24);
      if ( v8 == v11 )
        break;
      v13 = *(unsigned int *)(a1 + 56);
      v10 = *v11;
      a6 = v13 + 1;
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v25 = v8;
        v27 = v9;
        v29 = v11;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v13 + 1, 8u, v9, a6);
        v13 = *(unsigned int *)(a1 + 56);
        v8 = v25;
        v9 = v27;
        v11 = v29;
      }
      v12 = *(_QWORD *)(a1 + 48);
      ++v11;
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v15 = v33[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v15;
  if ( v15 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(v33);
  sub_2BF0340(a1 + 96, 1, a5, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v32);
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  v16 = *a5;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  if ( (unsigned __int8)(v16 - 82) <= 1u )
  {
    v17 = *((_WORD *)a5 + 1);
    *(_BYTE *)(a1 + 152) = 0;
    result = v17 & 0x3F;
    *(_DWORD *)(a1 + 156) = result;
    return result;
  }
  if ( (_BYTE)v16 == 58 )
  {
    *(_BYTE *)(a1 + 152) = 2;
LABEL_20:
    v19 = a5[1] >> 1;
LABEL_21:
    v20 = v19 & 1;
    result = v19 & 1 | *(_BYTE *)(a1 + 156) & 0xFEu;
    *(_BYTE *)(a1 + 156) = v20 | *(_BYTE *)(a1 + 156) & 0xFE;
    return result;
  }
  if ( (unsigned __int8)v16 <= 0x36u )
  {
    v21 = 0x40540000000000LL;
    if ( _bittest64(&v21, v16) )
    {
      v22 = a5[1];
      *(_BYTE *)(a1 + 152) = 1;
      result = *(_BYTE *)(a1 + 156) & 0xFC | (v22 >> 1) & 3u;
      *(_BYTE *)(a1 + 156) = result;
      return result;
    }
    if ( (unsigned int)(unsigned __int8)v16 - 48 <= 1 )
      goto LABEL_24;
  }
  else
  {
    if ( (unsigned __int8)(v16 - 55) <= 1u )
    {
LABEL_24:
      *(_BYTE *)(a1 + 152) = 3;
      goto LABEL_20;
    }
    if ( (_BYTE)v16 == 63 )
    {
      *(_BYTE *)(a1 + 152) = 4;
      result = sub_B4DE20((__int64)a5);
      *(_DWORD *)(a1 + 156) = result;
      return result;
    }
  }
  if ( (((_BYTE)v16 - 68) & 0xFB) == 0 )
  {
    *(_BYTE *)(a1 + 152) = 6;
    v19 = sub_B44910((__int64)a5);
    goto LABEL_21;
  }
  result = sub_920620((__int64)a5);
  if ( (_BYTE)result )
  {
    v23 = a5[1];
    *(_BYTE *)(a1 + 152) = 5;
    v24 = v23 >> 1;
    if ( v24 == 127 )
      v24 = -1;
    LODWORD(v33[0]) = v24;
    sub_2C1AC80(&v32, v33);
    result = (unsigned __int8)v32;
    *(_BYTE *)(a1 + 156) = v32;
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 7;
    *(_DWORD *)(a1 + 156) = 0;
  }
  return result;
}
