// Function: sub_BFBE00
// Address: 0xbfbe00
//
void __fastcall sub_BFBE00(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // esi
  unsigned int v6; // eax
  char v7; // dl
  __int64 v8; // r8
  int v9; // ecx
  char v10; // r9
  const char *v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  __int64 v14; // rax
  const char *v15; // [rsp+0h] [rbp-50h] BYREF
  char v16; // [rsp+20h] [rbp-30h]
  char v17; // [rsp+21h] [rbp-2Fh]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v5 = *(unsigned __int8 *)(v4 + 8);
  v6 = v5 - 17;
  v7 = *(_BYTE *)(v4 + 8);
  if ( (unsigned int)(v5 - 17) <= 1 )
    v7 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
  if ( v7 == 14 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    v9 = *(unsigned __int8 *)(v8 + 8);
    v10 = *(_BYTE *)(v8 + 8);
    if ( (unsigned int)(v9 - 17) <= 1 )
      v10 = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
    if ( v10 == 12 )
    {
      if ( (unsigned int)(v9 - 17) <= 1 == v6 <= 1 )
      {
        if ( v6 > 1 || *(_DWORD *)(v4 + 32) == *(_DWORD *)(v8 + 32) && ((_BYTE)v5 == 18) == ((_BYTE)v9 == 18) )
        {
          sub_BF6FE0(a1, a2);
          return;
        }
        v17 = 1;
        v11 = "PtrToInt Vector width mismatch";
      }
      else
      {
        v17 = 1;
        v11 = "PtrToInt type mismatch";
      }
    }
    else
    {
      v17 = 1;
      v11 = "PtrToInt result must be integral";
    }
  }
  else
  {
    v17 = 1;
    v11 = "PtrToInt source must be pointer";
  }
  v12 = *(_QWORD *)a1;
  v15 = v11;
  v16 = 3;
  if ( v12 )
  {
    sub_CA0E80(&v15, v12);
    v13 = *(_BYTE **)(v12 + 32);
    if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
    {
      sub_CB5D20(v12, 10);
    }
    else
    {
      *(_QWORD *)(v12 + 32) = v13 + 1;
      *v13 = 10;
    }
    v14 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( v14 )
      sub_BDBD80(a1, (_BYTE *)a2);
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
}
