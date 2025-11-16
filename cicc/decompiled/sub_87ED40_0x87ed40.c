// Function: sub_87ED40
// Address: 0x87ed40
//
_QWORD *__fastcall sub_87ED40(__int64 *a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r15
  int v8; // eax
  _QWORD *v9; // r12
  char v10; // al
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // edx
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h]

  v5 = 0;
  v8 = sub_7CF970();
  if ( v8 != -1 )
    v5 = qword_4F04C68[0] + 776LL * v8;
  v9 = sub_87ECE0(a1, (_QWORD *)(a2 + 8), v8);
  *((_BYTE *)v9 + 81) = *(_BYTE *)(a2 + 17) & 0x20 | *((_BYTE *)v9 + 81) & 0xDF;
  v10 = *(_BYTE *)(a2 + 16);
  *(_QWORD *)(a2 + 24) = v9;
  *(_BYTE *)(a2 + 16) = a3 & 1 | v10 & 0xFE;
  v11 = *v9;
  if ( a3 )
  {
    if ( a4 )
    {
      v16 = a4;
      if ( (*(_BYTE *)(a4 + 124) & 1) != 0 )
      {
        v17 = *v9;
        v16 = sub_735B70(a4);
        v11 = v17;
      }
      v12 = *(_QWORD *)(*(_QWORD *)v16 + 96LL);
    }
    else
    {
      v12 = *(_QWORD *)(qword_4F04C68[0] + 24LL);
      if ( !v12 )
        v12 = qword_4F04C68[0] + 32LL;
    }
  }
  else
  {
    v12 = *(_QWORD *)(v5 + 24);
    if ( !v12 )
      v12 = v5 + 32;
  }
  v13 = *(_QWORD *)(v12 + 8);
  v9[2] = v13;
  if ( v13 )
    *(_QWORD *)(v13 + 24) = v9;
  *(_QWORD *)(v12 + 8) = v9;
  v14 = a5 & 0xFFBFF468;
  if ( (a5 & 0xFFBFF468) == 0 )
  {
    v9[1] = *(_QWORD *)(v11 + 40);
    *(_QWORD *)(v11 + 40) = v9;
  }
  LOBYTE(v14) = v14 != 0;
  *((_WORD *)v9 + 41) = *((_WORD *)v9 + 41) & 0xFEE7 | ((_WORD)v14 << 8) | (16 * (a3 & 1)) | 8;
  if ( a3 && a4 )
  {
    sub_877E90((__int64)v9, 0, a4);
LABEL_16:
    *((_DWORD *)v9 + 10) = *(_DWORD *)(*(_QWORD *)(a4 + 128) + 24LL);
    goto LABEL_14;
  }
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 4) - 3) <= 1u )
  {
    sub_877E90((__int64)v9, 0, *(_QWORD *)(v5 + 224));
    if ( !a3 )
      goto LABEL_14;
    goto LABEL_22;
  }
  if ( a3 )
  {
LABEL_22:
    if ( !a4 )
    {
      *((_DWORD *)v9 + 10) = *(_DWORD *)qword_4F04C68[0];
      goto LABEL_14;
    }
    goto LABEL_16;
  }
LABEL_14:
  *((_WORD *)v9 + 41) = *((_WORD *)v9 + 41) & 0xF11F
                      | (((a5 >> 9) & 1) << 11)
                      | (((a5 >> 11) & 1) << 10)
                      | (((a5 >> 8) & 1) << 9)
                      | (((a5 >> 2) & 1) << 7)
                      | (32 * (a5 & 1))
                      | (((a5 >> 1) & 1) << 6);
  return v9;
}
