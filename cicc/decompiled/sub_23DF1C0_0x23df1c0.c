// Function: sub_23DF1C0
// Address: 0x23df1c0
//
__int64 __fastcall sub_23DF1C0(__int64 a1, _DWORD *a2, int a3, char a4)
{
  int v4; // r14d
  int v5; // ebx
  bool v6; // cl
  char v7; // r12
  bool v8; // al
  unsigned int v10; // ebx
  bool v11; // [rsp+6h] [rbp-4Ah]
  bool v12; // [rsp+7h] [rbp-49h]
  bool v14; // [rsp+Ch] [rbp-44h]
  char v15; // [rsp+Dh] [rbp-43h]
  bool v16; // [rsp+Eh] [rbp-42h]
  bool v17; // [rsp+10h] [rbp-40h]
  bool v18; // [rsp+11h] [rbp-3Fh]
  bool v19; // [rsp+12h] [rbp-3Eh]
  bool v20; // [rsp+13h] [rbp-3Dh]
  bool v22; // [rsp+18h] [rbp-38h]
  bool v23; // [rsp+19h] [rbp-37h]
  bool v24; // [rsp+1Ah] [rbp-36h]
  bool v25; // [rsp+1Bh] [rbp-35h]
  int v26; // [rsp+1Ch] [rbp-34h]

  v4 = a2[11];
  v5 = a2[8];
  v26 = a2[12];
  if ( v4 == 5 )
  {
    if ( v5 != 39 )
    {
      v16 = 1;
      v11 = 0;
      goto LABEL_5;
    }
    v16 = 1;
    v11 = 0;
    v8 = a2[10] == 3;
    goto LABEL_21;
  }
  v16 = v4 == 30 || (unsigned int)(v4 - 27) <= 1;
  if ( v4 != 1 )
  {
    v11 = v4 == 9;
    if ( v5 != 39 )
      goto LABEL_5;
    v8 = a2[10] == 3;
    v20 = v8 && v4 == 24;
    if ( v20 )
    {
LABEL_22:
      v15 = 1;
      v14 = v4 == 7;
      v6 = 0;
      v17 = (v26 & 0xFFFFFFEF) == 3;
      goto LABEL_23;
    }
LABEL_21:
    v20 = v8 && v4 == 25;
    goto LABEL_22;
  }
  if ( v5 == 39 )
  {
    v11 = 1;
    v8 = a2[10] == 3;
    goto LABEL_21;
  }
  v11 = 1;
LABEL_5:
  v14 = v4 == 7;
  v17 = (v26 & 0xFFFFFFEF) == 3;
  if ( v5 == 24 )
  {
    v20 = 0;
    v6 = 1;
    v15 = 0;
    v22 = 0;
    v19 = 0;
    goto LABEL_27;
  }
  v20 = 0;
  v15 = 0;
  v6 = v5 == 25;
  if ( v5 == 16 )
  {
    v22 = 0;
    v6 = 0;
    v19 = 1;
    v25 = 0;
    goto LABEL_8;
  }
LABEL_23:
  v19 = v5 == 17;
  if ( v5 == 18 )
  {
    v22 = 1;
    v7 = v6;
    v19 = 0;
    v24 = 0;
    v25 = 0;
    v18 = v4 == 14;
LABEL_25:
    v12 = v5 == 27;
    goto LABEL_10;
  }
  v22 = v5 == 19;
  v18 = v4 == 14;
  if ( v5 == 1 )
  {
    v22 = 0;
    v7 = v6;
    v24 = 0;
    v25 = 1;
    v12 = 0;
    goto LABEL_10;
  }
LABEL_27:
  v25 = v5 == 2 || (unsigned int)(v5 - 36) <= 1;
  v18 = v4 == 14;
  if ( v5 != 3 )
  {
LABEL_8:
    v24 = v5 == 4;
    v18 = v4 == 14;
    v7 = v6 || v5 == 4;
    if ( v5 == 26 )
    {
      v12 = 1;
      goto LABEL_10;
    }
    goto LABEL_25;
  }
  v24 = 1;
  v7 = 1;
  v12 = 0;
LABEL_10:
  v23 = v6;
  *(_DWORD *)a1 = 3;
  if ( (int)sub_23DF0D0(&dword_4FE0888) > 0 )
    *(_DWORD *)a1 = qword_4FE0908;
  if ( a3 == 32 )
  {
    if ( v26 == 17 )
      goto LABEL_69;
    if ( v17 )
      goto LABEL_65;
    if ( v19 )
    {
      *(_QWORD *)(a1 + 8) = 178913280;
      goto LABEL_43;
    }
    if ( v4 == 3 || v4 == 10 )
    {
      *(_QWORD *)(a1 + 8) = 0x40000000;
      goto LABEL_43;
    }
    if ( v16 )
      goto LABEL_69;
    if ( v4 == 14 )
    {
      *(_QWORD *)(a1 + 8) = 805306368;
      goto LABEL_43;
    }
    if ( v4 != 37 )
    {
LABEL_65:
      *(_QWORD *)(a1 + 8) = 0x20000000;
      goto LABEL_43;
    }
LABEL_42:
    *(_QWORD *)(a1 + 8) = 0;
    goto LABEL_43;
  }
  if ( v4 == 4 )
    goto LABEL_42;
  if ( v23 )
    goto LABEL_64;
  if ( v5 == 33 )
  {
    *(_QWORD *)(a1 + 8) = 0x10000000000000LL;
    goto LABEL_43;
  }
  if ( v4 == 3 && v24 )
  {
    *(_QWORD *)(a1 + 8) = 0x800000000000LL;
    goto LABEL_43;
  }
  if ( !v22 && v4 == 3 )
  {
    if ( a4 )
    {
      *(_QWORD *)(a1 + 8) = 0xDFFFF7C000000000LL;
      goto LABEL_43;
    }
    goto LABEL_80;
  }
  if ( v4 == 10 )
  {
    if ( a4 )
    {
      *(_QWORD *)(a1 + 8) = 0xDFFF900000000000LL;
      goto LABEL_43;
    }
    goto LABEL_80;
  }
  if ( v20 )
  {
    *(_QWORD *)(a1 + 8) = 0x10000000000LL;
    goto LABEL_43;
  }
  if ( v15 )
  {
    if ( v14 )
    {
      if ( a4 )
      {
        *(_QWORD *)(a1 + 8) = 0xDFFFFC0000000000LL;
        goto LABEL_43;
      }
      goto LABEL_88;
    }
    if ( v18 )
      goto LABEL_69;
  }
  if ( v22 )
  {
    *(_QWORD *)(a1 + 8) = 0x2000000000LL;
    goto LABEL_43;
  }
  if ( v16 )
    goto LABEL_69;
  if ( v11 )
  {
    if ( v24 )
      goto LABEL_69;
LABEL_61:
    if ( v5 != 14 )
    {
      if ( v5 != 29 )
      {
        if ( v12 )
        {
LABEL_88:
          *(_QWORD *)(a1 + 8) = (-4096LL << *(_DWORD *)a1) & 0x7FFFFFFF;
          goto LABEL_43;
        }
LABEL_64:
        *(_QWORD *)(a1 + 8) = 0x100000000000LL;
        goto LABEL_43;
      }
LABEL_69:
      *(_QWORD *)(a1 + 8) = -1;
      goto LABEL_43;
    }
LABEL_80:
    *(_QWORD *)(a1 + 8) = 0x400000000000LL;
    goto LABEL_43;
  }
  if ( !v24 )
    goto LABEL_61;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
LABEL_43:
  if ( byte_4FE1E88 )
    *(_QWORD *)(a1 + 8) = -1;
  if ( (int)sub_23DF0D0(&dword_4FE07A8) > 0 )
    *(_QWORD *)(a1 + 8) = qword_4FE0828;
  if ( v7 )
  {
    *(_BYTE *)(a1 + 16) = 0;
    if ( v26 != 17 )
    {
LABEL_52:
      v25 = 0;
      goto LABEL_53;
    }
  }
  else
  {
    if ( !v20 )
    {
      v7 = v5 != 14 && ((v5 - 29) & 0xFFFFFFFB) != 0;
      if ( v7 )
        v7 = *(_QWORD *)(a1 + 8) != -1 && (*(_QWORD *)(a1 + 8) & (*(_QWORD *)(a1 + 8) - 1LL)) == 0;
    }
    *(_BYTE *)(a1 + 16) = v7;
    if ( v26 != 17 )
      goto LABEL_52;
  }
  v10 = sub_CC7810((__int64)a2);
  if ( !sub_CC7F40((__int64)a2) && v10 <= 0x14 || !(_BYTE)qword_4FE1DA8 )
    goto LABEL_52;
LABEL_53:
  *(_BYTE *)(a1 + 17) = v25;
  return a1;
}
