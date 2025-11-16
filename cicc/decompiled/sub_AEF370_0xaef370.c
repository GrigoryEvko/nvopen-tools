// Function: sub_AEF370
// Address: 0xaef370
//
__int64 __fastcall sub_AEF370(__int64 **a1, _BYTE *a2)
{
  __int64 v2; // r12
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r8
  _BYTE *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // r8d
  __int64 *v17; // rsi
  __int64 v18; // rdx
  _BYTE *v19; // r12
  int v20; // r15d
  int v21; // eax
  __int64 v22; // rax
  int v24; // esi
  int v25; // r10d
  int v26; // edx
  int v27; // r9d
  int v28; // [rsp+8h] [rbp-48h]
  int v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (__int64)a2;
  if ( !a2 || *a2 != 6 )
    return v2;
  v3 = *a1;
  sub_B10CB0(&v30, a2);
  v4 = sub_B10D00(&v30);
  v5 = sub_B10D40(&v30);
  v6 = *v3;
  v7 = v5;
  if ( v4 )
  {
    sub_AEE3F0(*(_QWORD *)v6, v4);
    v8 = *(unsigned int *)(*(_QWORD *)v6 + 24LL);
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)v6 + 8LL);
      v10 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v4 == *v11 )
      {
LABEL_6:
        if ( v11 != (__int64 *)(v9 + 16 * v8) )
        {
          v13 = (_BYTE *)v11[1];
          if ( !v13 )
            goto LABEL_23;
          goto LABEL_8;
        }
      }
      else
      {
        v26 = 1;
        while ( v12 != -4096 )
        {
          v27 = v26 + 1;
          v10 = (v8 - 1) & (v26 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v4 == *v11 )
            goto LABEL_6;
          v26 = v27;
        }
      }
    }
    v13 = (_BYTE *)v4;
LABEL_8:
    if ( (unsigned __int8)(*v13 - 5) <= 0x1Fu )
    {
LABEL_9:
      **(_BYTE **)(v6 + 8) |= v4 != (_QWORD)v13;
      v6 = *v3;
      goto LABEL_10;
    }
LABEL_23:
    v13 = 0;
    goto LABEL_9;
  }
  LODWORD(v13) = 0;
LABEL_10:
  if ( !v7 )
  {
    LODWORD(v19) = 0;
    goto LABEL_17;
  }
  v28 = (int)v13;
  sub_AEE3F0(*(_QWORD *)v6, v7);
  LODWORD(v13) = v28;
  v14 = *(unsigned int *)(*(_QWORD *)v6 + 24LL);
  if ( !(_DWORD)v14 )
    goto LABEL_25;
  v15 = *(_QWORD *)(*(_QWORD *)v6 + 8LL);
  v16 = (v14 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v17 = (__int64 *)(v15 + 16LL * v16);
  v18 = *v17;
  if ( v7 != *v17 )
  {
    v24 = 1;
    while ( v18 != -4096 )
    {
      v25 = v24 + 1;
      v16 = (v14 - 1) & (v16 + v24);
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v7 == *v17 )
        goto LABEL_13;
      v24 = v25;
    }
    goto LABEL_25;
  }
LABEL_13:
  if ( v17 == (__int64 *)(v15 + 16 * v14) )
  {
LABEL_25:
    v19 = (_BYTE *)v7;
    goto LABEL_15;
  }
  v19 = (_BYTE *)v17[1];
  if ( !v19 )
  {
LABEL_22:
    v19 = 0;
    goto LABEL_16;
  }
LABEL_15:
  if ( (unsigned __int8)(*v19 - 5) > 0x1Fu )
    goto LABEL_22;
LABEL_16:
  **(_BYTE **)(v6 + 8) |= v7 != (_QWORD)v19;
LABEL_17:
  v29 = (int)v13;
  v20 = sub_B10CF0(&v30);
  v21 = sub_B10CE0(&v30);
  v22 = sub_B01860(*(_QWORD *)v3[1], v21, v20, v29, (_DWORD)v19, 0, 0, 1);
  sub_B10CB0(v31, v22);
  v2 = sub_B10CD0(v31);
  if ( v31[0] )
    sub_B91220(v31);
  if ( v30 )
    sub_B91220(&v30);
  return v2;
}
