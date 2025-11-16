// Function: sub_2D90B40
// Address: 0x2d90b40
//
__int64 __fastcall sub_2D90B40(__int64 a1, char *a2)
{
  char v4; // al
  size_t v5; // r8
  char *v6; // rsi
  _DWORD *v7; // rdx
  __int64 v8; // rax
  _BYTE *v9; // rdx
  _QWORD *v10; // rdi
  char v11; // al
  bool v12; // cc
  size_t v13; // rdx
  char *v14; // rsi
  _DWORD *v15; // rcx
  unsigned __int64 v17; // r9
  char *v18; // rcx
  char *v19; // rsi
  unsigned int v20; // eax
  unsigned int v21; // ecx
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  char *v24; // rdx
  char *v25; // rsi
  unsigned int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rcx
  _DWORD *v29; // rdx
  _QWORD v30[3]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v31; // [rsp+18h] [rbp-48h]
  _DWORD *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]
  __int64 v34; // [rsp+30h] [rbp-30h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v33 = 0x100000000LL;
  v34 = a1;
  v30[0] = &unk_49DD210;
  v30[1] = 0;
  v30[2] = 0;
  v31 = 0;
  v32 = 0;
  sub_CB5980((__int64)v30, 0, 0, 0);
  v4 = *a2;
  if ( *a2 == 2 )
  {
    v5 = 13;
    v6 = "positive-zero";
    goto LABEL_6;
  }
  if ( v4 > 2 )
  {
    v5 = 7;
    v6 = "dynamic";
    if ( v4 != 3 )
      goto LABEL_25;
LABEL_6:
    v7 = v32;
    if ( v5 > v31 - (unsigned __int64)v32 )
    {
LABEL_7:
      v8 = sub_CB6200((__int64)v30, (unsigned __int8 *)v6, v5);
      v9 = *(_BYTE **)(v8 + 32);
      v10 = (_QWORD *)v8;
      goto LABEL_8;
    }
LABEL_29:
    if ( (unsigned int)v5 < 8 )
    {
      *v7 = *(_DWORD *)v6;
      *(_DWORD *)((char *)v7 + (unsigned int)v5 - 4) = *(_DWORD *)&v6[(unsigned int)v5 - 4];
      v29 = v32;
    }
    else
    {
      v23 = (unsigned __int64)(v7 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v7 = *(_QWORD *)v6;
      *(_QWORD *)((char *)v7 + v5 - 8) = *(_QWORD *)&v6[v5 - 8];
      v24 = (char *)v7 - v23;
      v25 = (char *)(v6 - v24);
      if ( (((_DWORD)v5 + (_DWORD)v24) & 0xFFFFFFF8) >= 8 )
      {
        v26 = (v5 + (_DWORD)v24) & 0xFFFFFFF8;
        v27 = 0;
        do
        {
          v28 = v27;
          v27 += 8;
          *(_QWORD *)(v23 + v28) = *(_QWORD *)&v25[v28];
        }
        while ( v27 < v26 );
      }
      v29 = v32;
    }
    v9 = (char *)v29 + v5;
    v10 = v30;
    v32 = v9;
LABEL_8:
    if ( v10[3] > (unsigned __int64)v9 )
      goto LABEL_9;
    goto LABEL_26;
  }
  if ( !v4 )
  {
    v7 = v32;
    v5 = 4;
    v6 = "ieee";
    if ( v31 - (unsigned __int64)v32 < 4 )
      goto LABEL_7;
    goto LABEL_29;
  }
  if ( v4 == 1 )
  {
    v5 = 13;
    v6 = "preserve-sign";
    goto LABEL_6;
  }
LABEL_25:
  v10 = v30;
  v9 = v32;
  if ( v31 > (unsigned __int64)v32 )
  {
LABEL_9:
    v10[4] = v9 + 1;
    *v9 = 44;
    v11 = a2[1];
    v12 = v11 <= 2;
    if ( v11 != 2 )
      goto LABEL_10;
LABEL_27:
    v13 = 13;
    v14 = "positive-zero";
LABEL_14:
    v15 = (_DWORD *)v10[4];
    if ( v10[3] - (_QWORD)v15 < v13 )
    {
LABEL_15:
      sub_CB6200((__int64)v10, (unsigned __int8 *)v14, v13);
      goto LABEL_16;
    }
    goto LABEL_19;
  }
LABEL_26:
  v10 = (_QWORD *)sub_CB5D20((__int64)v10, 44);
  v11 = a2[1];
  v12 = v11 <= 2;
  if ( v11 == 2 )
    goto LABEL_27;
LABEL_10:
  if ( v12 )
  {
    if ( v11 )
    {
      if ( v11 != 1 )
        goto LABEL_16;
      v13 = 13;
      v14 = "preserve-sign";
    }
    else
    {
      v13 = 4;
      v14 = "ieee";
    }
    goto LABEL_14;
  }
  v13 = 7;
  v14 = "dynamic";
  if ( v11 != 3 )
    goto LABEL_16;
  v15 = (_DWORD *)v10[4];
  if ( v10[3] - (_QWORD)v15 < 7u )
    goto LABEL_15;
LABEL_19:
  if ( (unsigned int)v13 < 8 )
  {
    *v15 = *(_DWORD *)v14;
    *(_DWORD *)((char *)v15 + (unsigned int)v13 - 4) = *(_DWORD *)&v14[(unsigned int)v13 - 4];
  }
  else
  {
    v17 = (unsigned __int64)(v15 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v15 = *(_QWORD *)v14;
    *(_QWORD *)((char *)v15 + v13 - 8) = *(_QWORD *)&v14[v13 - 8];
    v18 = (char *)v15 - v17;
    v19 = (char *)(v14 - v18);
    if ( (((_DWORD)v13 + (_DWORD)v18) & 0xFFFFFFF8) >= 8 )
    {
      v20 = (v13 + (_DWORD)v18) & 0xFFFFFFF8;
      v21 = 0;
      do
      {
        v22 = v21;
        v21 += 8;
        *(_QWORD *)(v17 + v22) = *(_QWORD *)&v19[v22];
      }
      while ( v21 < v20 );
    }
  }
  v10[4] += v13;
LABEL_16:
  v30[0] = &unk_49DD210;
  sub_CB5840((__int64)v30);
  return a1;
}
