// Function: sub_8EF310
// Address: 0x8ef310
//
__int64 __fastcall sub_8EF310(__int64 a1, unsigned __int8 *a2)
{
  _BYTE *v2; // r10
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // ecx
  int v7; // edi
  bool v8; // zf
  __int64 v9; // rcx
  __int64 v10; // r8
  char v11; // r8
  char v12; // al
  unsigned __int8 v13; // dl
  unsigned __int8 v14; // dl
  int v15; // ecx
  __int64 v16; // rax
  int v17; // eax
  __int64 result; // rax
  unsigned __int8 v19; // dl
  _BYTE *v20; // rdi
  int v21; // r12d

  v2 = (_BYTE *)(a1 + 12);
  v3 = 0;
  v4 = 15;
  v5 = unk_4F07580;
  *(_BYTE *)(a1 + 26) = 0;
  v7 = 0;
  do
  {
    v8 = v5 == 0;
    v9 = v3;
    v10 = v4;
    if ( v8 )
      v9 = v4;
    *(_BYTE *)(a1 + v3 + 12) = a2[v9];
    v5 = unk_4F07580;
    if ( unk_4F07580 )
      v10 = v3;
    if ( a2[v10] )
      v7 = 1;
    ++v3;
    --v4;
  }
  while ( v3 != 13 );
  v11 = *(_BYTE *)(a1 + 26);
  v12 = v11 | 1;
  if ( unk_4F07580 )
  {
    v13 = a2[13];
    *(_BYTE *)(a1 + 26) = v12;
    *(_BYTE *)(a1 + 25) = v13;
    v14 = a2[15];
    if ( a2[13] )
      v7 = 1;
    v15 = (v14 & 0x7F) << 8;
    v16 = 14;
  }
  else
  {
    v19 = a2[2];
    *(_BYTE *)(a1 + 26) = v12;
    *(_BYTE *)(a1 + 25) = v19;
    v14 = *a2;
    if ( a2[2] )
      v7 = 1;
    v15 = (v14 & 0x7F) << 8;
    v16 = 1;
  }
  v17 = v15 | a2[v16];
  *(_DWORD *)(a1 + 4) = v14 >> 7;
  if ( v17 == 0x7FFF )
  {
    result = 16385;
    if ( v7 )
      *(_DWORD *)a1 = 3;
    else
      *(_DWORD *)a1 = 4;
    *(_DWORD *)(a1 + 8) = 16385;
    *(_DWORD *)(a1 + 28) = 113;
  }
  else if ( v17 )
  {
    result = (unsigned int)(v17 - 16382);
    *(_DWORD *)a1 = 2;
    *(_DWORD *)(a1 + 8) = result;
    *(_DWORD *)(a1 + 28) = 113;
  }
  else if ( v7 )
  {
    v20 = v2;
    *(_BYTE *)(a1 + 26) = v11 & 0xFE;
    v21 = sub_8EE4D0(v2, 113);
    sub_8EE880(v20, 113, v21);
    *(_DWORD *)a1 = 2;
    result = (unsigned int)(-16381 - v21);
    *(_DWORD *)(a1 + 28) = 113;
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    *(_DWORD *)a1 = 6;
    *(_DWORD *)(a1 + 8) = -16382;
    *(_DWORD *)(a1 + 28) = 113;
    return 4294950914LL;
  }
  return result;
}
