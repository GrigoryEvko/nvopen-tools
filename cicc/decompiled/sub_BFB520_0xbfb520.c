// Function: sub_BFB520
// Address: 0xbfb520
//
void __fastcall sub_BFB520(__int64 a1, __int64 a2)
{
  __int64 v3; // r10
  __int64 v4; // r9
  int v5; // esi
  int v6; // r8d
  unsigned int v7; // eax
  unsigned __int8 v8; // cl
  const char *v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  const char *v13; // [rsp+0h] [rbp-50h] BYREF
  char v14; // [rsp+20h] [rbp-30h]
  char v15; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v5 = *(unsigned __int8 *)(v3 + 8);
  v6 = *(unsigned __int8 *)(v4 + 8);
  v7 = v6 - 17;
  if ( (unsigned int)(v6 - 17) <= 1 == (unsigned int)(v5 - 17) <= 1 )
  {
    v8 = *(_BYTE *)(v4 + 8);
    if ( v7 <= 1 )
      v8 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
    if ( v8 > 3u && v8 != 5 && (v8 & 0xFD) != 4 )
    {
      v15 = 1;
      v9 = "FPToSI source must be FP or FP vector";
      goto LABEL_10;
    }
    if ( (unsigned int)(v5 - 17) > 1 )
    {
      if ( (_BYTE)v5 == 12 )
      {
        if ( v7 > 1 )
          goto LABEL_21;
        goto LABEL_7;
      }
    }
    else if ( *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL) == 12 )
    {
LABEL_7:
      if ( ((_BYTE)v6 == 18) != ((_BYTE)v5 == 18) || *(_DWORD *)(v4 + 32) != *(_DWORD *)(v3 + 32) )
      {
        v15 = 1;
        v9 = "FPToSI source and dest vector length mismatch";
        goto LABEL_10;
      }
LABEL_21:
      sub_BF6FE0(a1, a2);
      return;
    }
    v15 = 1;
    v9 = "FPToSI result must be integer or integer vector";
  }
  else
  {
    v15 = 1;
    v9 = "FPToSI source and dest must both be vector or scalar";
  }
LABEL_10:
  v10 = *(_QWORD *)a1;
  v13 = v9;
  v14 = 3;
  if ( v10 )
  {
    sub_CA0E80(&v13, v10);
    v11 = *(_BYTE **)(v10 + 32);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
    {
      sub_CB5D20(v10, 10);
    }
    else
    {
      *(_QWORD *)(v10 + 32) = v11 + 1;
      *v11 = 10;
    }
    v12 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( v12 )
      sub_BDBD80(a1, (_BYTE *)a2);
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
}
