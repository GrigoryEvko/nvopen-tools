// Function: sub_794F10
// Address: 0x794f10
//
__int64 __fastcall sub_794F10(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  char v6; // dl
  int v7; // edx
  __int64 result; // rax
  _QWORD **v9; // r8
  unsigned int v10; // eax
  unsigned int v11; // r12d
  __int64 v12; // r15
  size_t v13; // r9
  __int64 v14; // r12
  char *v15; // rcx
  char *v16; // r15
  int v17; // eax
  unsigned int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD **v21; // [rsp+8h] [rbp-48h]
  size_t v22; // [rsp+8h] [rbp-48h]
  size_t v23; // [rsp+8h] [rbp-48h]
  unsigned int v24[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v24[0] = 1;
  if ( (*(_BYTE *)(a2 + 25) & 3) != 0 || (v6 = *(_BYTE *)(a3 + 140), v6 == 6) )
  {
    if ( (unsigned int)sub_786210(a1, (_QWORD **)a2, a4, (char *)a4) )
      goto LABEL_19;
    return 0;
  }
  if ( (unsigned __int8)(v6 - 9) > 2u )
  {
    v24[0] = 0;
    result = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xAA1u, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v6 = *(_BYTE *)(a3 + 140);
      result = v24[0];
    }
    if ( !v6 )
      *(_BYTE *)(a1 + 132) |= 0x40u;
    goto LABEL_20;
  }
  v7 = sub_7764B0(a1, a3, v24);
  result = v24[0];
  if ( !v24[0] )
    return result;
  v9 = (_QWORD **)a2;
  if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 8) > 3u )
  {
    v13 = 8;
    v12 = 16;
    v11 = 16;
  }
  else
  {
    v10 = (unsigned int)(v7 + 7) >> 3;
    v11 = v10 + 9;
    if ( (((_BYTE)v10 + 9) & 7) != 0 )
      v11 = v10 + 17 - (((_BYTE)v10 + 9) & 7);
    v12 = v11;
    v13 = v11 - 8LL;
  }
  v14 = v7 + v11;
  if ( (unsigned int)v14 > 0x400 )
  {
    v22 = v13;
    v18 = v14 + 16;
    v19 = sub_822B10(v18);
    v20 = *(_QWORD *)(a1 + 32);
    v13 = v22;
    *(_DWORD *)(v19 + 8) = v18;
    v9 = (_QWORD **)a2;
    v15 = (char *)(v19 + 16);
    *(_QWORD *)v19 = v20;
    *(_DWORD *)(v19 + 12) = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v19;
  }
  else
  {
    v15 = *(char **)(a1 + 16);
    if ( (v14 & 7) != 0 )
      v14 = (_DWORD)v14 + 8 - (unsigned int)(v14 & 7);
    if ( 0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24)) < (unsigned int)v14 )
    {
      v23 = v13;
      sub_772E70((_QWORD *)(a1 + 16));
      v15 = *(char **)(a1 + 16);
      v9 = (_QWORD **)a2;
      v13 = v23;
    }
    *(_QWORD *)(a1 + 16) = &v15[v14];
  }
  v21 = v9;
  v16 = (char *)memset(v15, 0, v13) + v12;
  *((_QWORD *)v16 - 1) = a3;
  if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) <= 2u )
    *(_QWORD *)v16 = 0;
  if ( !(unsigned int)sub_786210(a1, v21, (unsigned __int64)v16, v16) )
    return 0;
  *(v16 - 9) |= 1u;
  *(_QWORD *)a4 = v16;
  *(_DWORD *)(a4 + 8) = 0;
  v17 = *(_DWORD *)(a1 + 40);
  *(_QWORD *)(a4 + 24) = v16;
  *(_DWORD *)(a4 + 12) = v17;
LABEL_19:
  result = v24[0];
LABEL_20:
  *(_BYTE *)(a4 - 9) |= 1u;
  return result;
}
