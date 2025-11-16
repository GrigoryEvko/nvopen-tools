// Function: sub_180D640
// Address: 0x180d640
//
__int64 __fastcall sub_180D640(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  int v11; // edx
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  char v14; // r8
  __int64 v15; // rax
  int v16; // edi
  unsigned int v17; // esi
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // r9d
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r15
  unsigned __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a1 + 792);
  if ( !(_DWORD)v5 )
    goto LABEL_8;
  v6 = *(_QWORD *)(a1 + 776);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v11 = 1;
    while ( v9 != -8 )
    {
      v21 = v11 + 1;
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v11 = v21;
    }
LABEL_8:
    v12 = *(_QWORD *)(a2 + 56);
    v13 = *(unsigned __int8 *)(v12 + 8);
    if ( (unsigned __int8)v13 > 0xFu || (v20 = 35454, !_bittest64(&v20, v13)) )
    {
      if ( (unsigned int)(v13 - 13) > 1 && (_DWORD)v13 != 16 || !sub_16435F0(v12, 0) )
        goto LABEL_12;
    }
    if ( (unsigned __int8)sub_15F8F00(a2) )
    {
      if ( (unsigned __int8)sub_15F8BF0(a2) )
      {
        v22 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v22 + 16) != 13 )
          BUG();
        v28 = *(_DWORD *)(v22 + 32) <= 0x40u ? *(_QWORD *)(v22 + 24) : **(_QWORD **)(v22 + 24);
      }
      else
      {
        v28 = 1;
      }
      v23 = *(_QWORD *)(a2 + 56);
      v24 = sub_15F2050(a2);
      v25 = sub_1632FA0(v24);
      v26 = (unsigned int)sub_15A9FE0(v25, v23);
      v27 = sub_127FA20(v25, v23);
      v2 = v26 * v28;
      if ( !(v26 * v28 * ((v26 + ((unsigned __int64)(v27 + 7) >> 3) - 1) / v26)) )
        goto LABEL_12;
    }
    if ( !byte_4FA7AC0 || !(unsigned __int8)sub_1B33710(a2, 0) )
      LOBYTE(v2) = (*(_BYTE *)(a2 + 18) & 0x60) == 0;
    else
LABEL_12:
      v2 = 0;
    v29 = a2;
    v14 = sub_180D270(a1 + 768, &v29, v30);
    v15 = v30[0];
    if ( v14 )
      goto LABEL_19;
    v16 = *(_DWORD *)(a1 + 784);
    v17 = *(_DWORD *)(a1 + 792);
    ++*(_QWORD *)(a1 + 768);
    v18 = v16 + 1;
    if ( 4 * (v16 + 1) >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a1 + 788) - v18 > v17 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 784) = v18;
      if ( *(_QWORD *)v15 != -8 )
        --*(_DWORD *)(a1 + 788);
      v19 = v29;
      *(_BYTE *)(v15 + 8) = 0;
      *(_QWORD *)v15 = v19;
LABEL_19:
      *(_BYTE *)(v15 + 8) = v2;
      return v2;
    }
    sub_180D480(a1 + 768, v17);
    sub_180D270(a1 + 768, &v29, v30);
    v15 = v30[0];
    v18 = *(_DWORD *)(a1 + 784) + 1;
    goto LABEL_16;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v6 + 16 * v5) )
    goto LABEL_8;
  return *((unsigned __int8 *)v8 + 8);
}
