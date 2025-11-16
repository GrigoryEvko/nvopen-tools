// Function: sub_BFB6E0
// Address: 0xbfb6e0
//
void __fastcall sub_BFB6E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r8
  __int64 v5; // rdi
  int v6; // ecx
  int v7; // esi
  char v8; // r9
  unsigned __int8 v9; // al
  const char *v10; // rax
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rax
  _QWORD v14[4]; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v6 = *(unsigned __int8 *)(v4 + 8);
  v7 = *(unsigned __int8 *)(v5 + 8);
  if ( (unsigned int)(v6 - 17) <= 1 == (unsigned int)(v7 - 17) <= 1 )
  {
    v8 = *(_BYTE *)(v5 + 8);
    if ( (unsigned int)(v7 - 17) <= 1 )
      v8 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
    if ( v8 == 12 )
    {
      if ( (unsigned int)(v6 - 17) > 1 )
      {
        if ( (unsigned __int8)v6 > 3u && (_BYTE)v6 != 5 && (v6 & 0xFD) != 4 )
        {
LABEL_19:
          v16 = 1;
          v14[0] = "UIToFP result must be FP or FP vector";
          v15 = 3;
          sub_BDBF70((__int64 *)a1, (__int64)v14);
          if ( !*(_QWORD *)a1 )
            return;
LABEL_15:
          sub_BDBD80(a1, (_BYTE *)a2);
          return;
        }
      }
      else
      {
        v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
        if ( v9 > 3u && v9 != 5 && (v9 & 0xFD) != 4 )
          goto LABEL_19;
        if ( *(_DWORD *)(v5 + 32) != *(_DWORD *)(v4 + 32) || ((_BYTE)v7 == 18) != ((_BYTE)v6 == 18) )
        {
          v16 = 1;
          v10 = "UIToFP source and dest vector length mismatch";
          goto LABEL_11;
        }
      }
      sub_BF6FE0(a1, a2);
      return;
    }
    v16 = 1;
    v10 = "UIToFP source must be integer or integer vector";
  }
  else
  {
    v16 = 1;
    v10 = "UIToFP source and dest must both be vector or scalar";
  }
LABEL_11:
  v11 = *(_QWORD *)a1;
  v14[0] = v10;
  v15 = 3;
  if ( !v11 )
  {
    *(_BYTE *)(a1 + 152) = 1;
    return;
  }
  sub_CA0E80(v14, v11);
  v12 = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
  {
    sub_CB5D20(v11, 10);
  }
  else
  {
    *(_QWORD *)(v11 + 32) = v12 + 1;
    *v12 = 10;
  }
  v13 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v13 )
    goto LABEL_15;
}
