// Function: sub_12FCB00
// Address: 0x12fcb00
//
__int64 __fastcall sub_12FCB00(__int64 a1, unsigned __int64 a2)
{
  _BYTE *v2; // r10
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 *v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rax
  unsigned __int64 v11; // rdx
  _BYTE *v12; // r15
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // rcx
  _QWORD *v15; // r12
  _QWORD *v16; // rdx
  unsigned int i; // esi
  _BYTE *v18; // rdi
  unsigned __int64 v19; // rax
  __int64 j; // rax
  int v21; // esi
  _BYTE *v22; // rdi
  _BYTE *v23; // rdx
  _BYTE v24[432]; // [rsp+0h] [rbp-1B0h] BYREF

  v2 = (_BYTE *)(a1 + 432);
  if ( !a1 )
  {
    sub_130D500(v24);
    v2 = v24;
  }
  v5 = a2 & 0xFFFFFFFFC0000000LL;
  v6 = (a2 >> 26) & 0xF0;
  v7 = (unsigned __int64 *)&v2[v6];
  v8 = *(_QWORD *)&v2[v6];
  if ( (a2 & 0xFFFFFFFFC0000000LL) == v8 )
  {
    v9 = (_QWORD *)(v7[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v5 == *((_QWORD *)v2 + 32) )
  {
    v11 = *((_QWORD *)v2 + 33);
LABEL_9:
    *((_QWORD *)v2 + 32) = v8;
    *((_QWORD *)v2 + 33) = v7[1];
    *v7 = v5;
    v7[1] = v11;
    v9 = (_QWORD *)(v11 + ((a2 >> 9) & 0x1FFFF8));
  }
  else
  {
    v16 = v2 + 272;
    for ( i = 1; i != 8; ++i )
    {
      if ( v5 == *v16 )
      {
        v18 = &v2[16 * i];
        v2 += 16 * i - 16;
        v11 = *((_QWORD *)v18 + 33);
        *((_QWORD *)v18 + 32) = *((_QWORD *)v2 + 32);
        *((_QWORD *)v18 + 33) = *((_QWORD *)v2 + 33);
        goto LABEL_9;
      }
      v16 += 2;
    }
    v9 = (_QWORD *)sub_130D370(a1, &unk_5060AE0, v2, a2, 1, 0);
  }
  if ( (*v9 & 1) != 0 )
    return sub_1315B20(a1, a2);
  v12 = (_BYTE *)(a1 + 432);
  if ( !a1 )
  {
    v12 = v24;
    sub_130D500(v24);
  }
  v13 = (unsigned __int64 *)&v12[v6];
  v14 = *v13;
  if ( v5 == *v13 )
  {
    v15 = (_QWORD *)(v13[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v5 == *((_QWORD *)v12 + 32) )
  {
    *((_QWORD *)v12 + 32) = v14;
    v19 = *((_QWORD *)v12 + 33);
    *((_QWORD *)v12 + 33) = v13[1];
LABEL_20:
    *v13 = v5;
    v13[1] = v19;
    v15 = (_QWORD *)(v19 + ((a2 >> 9) & 0x1FFFF8));
  }
  else
  {
    for ( j = 1; j != 8; ++j )
    {
      v21 = j;
      if ( v5 == *(_QWORD *)&v12[16 * j + 256] )
      {
        v22 = &v12[16 * j];
        v19 = *((_QWORD *)v22 + 33);
        v23 = &v12[16 * (v21 - 1)];
        *((_QWORD *)v22 + 32) = *((_QWORD *)v23 + 32);
        *((_QWORD *)v22 + 33) = *((_QWORD *)v23 + 33);
        *((_QWORD *)v23 + 32) = v14;
        *((_QWORD *)v23 + 33) = v13[1];
        goto LABEL_20;
      }
    }
    v15 = (_QWORD *)sub_130D370(a1, &unk_5060AE0, v12, a2, 1, 0);
  }
  return sub_130A160(a1, ((__int64)(*v15 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
}
