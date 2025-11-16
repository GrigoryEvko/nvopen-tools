// Function: sub_1342130
// Address: 0x1342130
//
void __fastcall sub_1342130(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 v6; // r13
  _QWORD *v9; // rdx
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // r10
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r8
  _QWORD *v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // r8
  _QWORD *v22; // rcx
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // rsi
  _QWORD *v25; // r11
  unsigned int i; // r9d
  _QWORD *v27; // r11
  _QWORD *v28; // r9
  _QWORD *v29; // r9
  unsigned int j; // r15d
  _QWORD *v31; // r9
  _QWORD *v32; // rdx
  unsigned __int64 v33; // r11
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  bool v36; // [rsp+Fh] [rbp-1C1h]
  unsigned __int64 v37; // [rsp+10h] [rbp-1C0h]
  _QWORD *v38; // [rsp+10h] [rbp-1C0h]
  char v39; // [rsp+18h] [rbp-1B8h]
  _QWORD v40[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v6 = a4;
  v9 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    sub_130D500(v40);
    v9 = v40;
  }
  if ( (_DWORD)v6 != 232 )
  {
    v10 = *(_QWORD *)(a3 + 8);
    v11 = v10 & 0xFFFFFFFFC0000000LL;
    v12 = (*(_QWORD *)a3 >> 17) & 7LL;
    v13 = (*(_QWORD *)a3 >> 44) & 1LL;
    v14 = (_QWORD *)((char *)v9 + ((v10 >> 26) & 0xF0));
    v15 = *v14;
    if ( (v10 & 0xFFFFFFFFC0000000LL) == *v14 )
    {
      v16 = (_QWORD *)(v14[1] + ((v10 >> 9) & 0x1FFFF8));
    }
    else if ( v11 == v9[32] )
    {
      v23 = v9[33];
      v9[32] = v15;
      v16 = (_QWORD *)(v23 + ((v10 >> 9) & 0x1FFFF8));
      v9[33] = v14[1];
      *v14 = v11;
      v14[1] = v23;
    }
    else
    {
      v25 = v9 + 34;
      for ( i = 1; i != 8; ++i )
      {
        if ( v11 == *v25 )
        {
          v27 = &v9[2 * i];
          v37 = v27[33];
          v28 = &v9[2 * i - 2];
          v27[32] = v28[32];
          v27[33] = v28[33];
          v28[32] = v15;
          v28[33] = v14[1];
          *v14 = v11;
          v14[1] = v37;
          v16 = (_QWORD *)(v37 + ((v10 >> 9) & 0x1FFFF8));
          goto LABEL_6;
        }
        v25 += 2;
      }
      v36 = (*(_QWORD *)a3 & 0x100000000000LL) != 0;
      v38 = v9;
      v34 = sub_130D370(a1, a2, v9, v10, 0, 1);
      LOBYTE(v13) = v36;
      v9 = v38;
      v16 = (_QWORD *)v34;
    }
LABEL_6:
    if ( v16 )
      *v16 = (2LL * (unsigned __int8)v13) | (4 * v12) | (v6 << 48) | a5 | a3 & 0xFFFFFFFFFFFFLL;
    if ( a5 )
    {
      v17 = *(_QWORD *)(a3 + 16) & 0xFFFFFFFFFFFFF000LL;
      if ( v17 > 0x1000 )
      {
        v18 = v17 + (*(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFF000LL) - 4096;
        v19 = v18 & 0xFFFFFFFFC0000000LL;
        v20 = (_QWORD *)((char *)v9 + ((v18 >> 26) & 0xF0));
        v21 = *v20;
        if ( (v18 & 0xFFFFFFFFC0000000LL) == *v20 )
        {
          v22 = (_QWORD *)(v20[1] + ((v18 >> 9) & 0x1FFFF8));
        }
        else if ( v19 == v9[32] )
        {
          v24 = v9[33];
          v9[32] = v21;
          v22 = (_QWORD *)(v24 + ((v18 >> 9) & 0x1FFFF8));
          v9[33] = v20[1];
          *v20 = v19;
          v20[1] = v24;
        }
        else
        {
          v29 = v9 + 34;
          for ( j = 1; j != 8; ++j )
          {
            if ( v19 == *v29 )
            {
              v31 = &v9[2 * j];
              v32 = &v9[2 * j - 2];
              v33 = v31[33];
              v31[32] = v32[32];
              v22 = (_QWORD *)(v33 + ((v18 >> 9) & 0x1FFFF8));
              v31[33] = v32[33];
              v32[32] = v21;
              v32[33] = v20[1];
              *v20 = v19;
              v20[1] = v33;
              goto LABEL_12;
            }
            v29 += 2;
          }
          v39 = v13;
          v35 = sub_130D370(a1, a2, v9, v18, 0, 1);
          LOBYTE(v13) = v39;
          v22 = (_QWORD *)v35;
        }
LABEL_12:
        if ( v22 )
          *v22 = (2LL * (unsigned __int8)v13) | a3 & 0xFFFFFFFFFFFFLL | (v6 << 48) | (4 * v12) | 1;
      }
    }
  }
}
