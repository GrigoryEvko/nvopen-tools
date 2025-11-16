// Function: sub_777680
// Address: 0x777680
//
unsigned __int64 __fastcall sub_777680(__int64 a1, unsigned __int64 a2, __int64 *a3, _DWORD *a4)
{
  int v4; // r8d
  unsigned __int64 result; // rax
  unsigned int v10; // eax
  unsigned int v11; // ebx
  unsigned int v12; // ebx
  size_t v13; // rdx
  char *v14; // rdi
  size_t v15; // rbx
  _QWORD *v16; // rdi
  unsigned int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rax
  unsigned int v22; // [rsp+0h] [rbp-40h]
  unsigned int v23; // [rsp+0h] [rbp-40h]
  unsigned int v24; // [rsp+4h] [rbp-3Ch]
  int v25; // [rsp+4h] [rbp-3Ch]
  int v26; // [rsp+4h] [rbp-3Ch]
  int v27; // [rsp+4h] [rbp-3Ch]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v4 = 16;
  result = (unsigned int)*(unsigned __int8 *)(a2 + 140) - 2;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 2) > 1u )
  {
    result = sub_7764B0(a1, a2, a4);
    v4 = result;
  }
  if ( *a4 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 8) > 3u )
    {
      v28 = 16;
      v11 = 16;
    }
    else
    {
      v10 = (unsigned int)(v4 + 7) >> 3;
      v11 = v10 + 9;
      if ( (((_BYTE)v10 + 9) & 7) != 0 )
        v11 = v10 + 17 - (((_BYTE)v10 + 9) & 7);
      v28 = v11;
    }
    if ( (v4 & 7) != 0 )
      v4 = v4 + 8 - (v4 & 7);
    v12 = v4 + v11;
    v13 = v12 + 16;
    if ( (*(_BYTE *)(a1 + 132) & 8) == 0 )
    {
      v20 = (_QWORD *)qword_4F082A0;
      if ( qword_4F082A0 )
      {
        qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
      }
      else
      {
        v27 = v4;
        v20 = (_QWORD *)sub_823970(0x10000);
        v4 = v27;
        v13 = v12 + 16;
      }
      *v20 = *(_QWORD *)(a1 + 152);
      *(_QWORD *)(a1 + 152) = v20;
      v20[1] = 0;
      v21 = *(_QWORD *)(a1 + 152);
      *(_BYTE *)(a1 + 132) |= 8u;
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 144) = v21 + 24;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 168) = 0;
    }
    if ( (unsigned int)v13 > 0x400 )
    {
      v22 = v13;
      v17 = v12 + 32;
      v25 = v4;
      v18 = sub_822B10(v17);
      v19 = *(_QWORD *)(a1 + 160);
      v4 = v25;
      *(_DWORD *)(v18 + 8) = v17;
      v13 = v22;
      v14 = (char *)(v18 + 16);
      *(_QWORD *)v18 = v19;
      *(_DWORD *)(v18 + 12) = *(_DWORD *)(a1 + 168);
      *(_QWORD *)(a1 + 160) = v18;
    }
    else
    {
      v14 = *(char **)(a1 + 144);
      v15 = v12 + 24 - (v13 & 7);
      if ( (v13 & 7) == 0 )
        v15 = v13;
      if ( 0x10000 - (*(_DWORD *)(a1 + 144) - *(_DWORD *)(a1 + 152)) < (unsigned int)v15 )
      {
        v23 = v13;
        v26 = v4;
        sub_772E70((_QWORD *)(a1 + 144));
        v14 = *(char **)(a1 + 144);
        v13 = v23;
        v4 = v26;
      }
      *(_QWORD *)(a1 + 144) = &v14[v15];
    }
    v24 = v4;
    v16 = (char *)memset(v14, 0, v13) + v28;
    *(_DWORD *)((char *)v16 + v24) = 0;
    *(v16 - 1) = a2;
    *a3 = (__int64)v16;
    if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) <= 2u )
      *v16 = 0;
    result = (unsigned int)*a4;
    if ( (_DWORD)result )
      return sub_7745A0(a1, *a3);
  }
  return result;
}
