// Function: sub_8EF190
// Address: 0x8ef190
//
__int64 __fastcall sub_8EF190(__int64 a1, unsigned __int8 *a2)
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
  unsigned __int8 v12; // dl
  int v13; // ecx
  __int64 v14; // rax
  int v15; // eax
  int v16; // eax
  __int64 result; // rax
  _BYTE *v18; // rdi
  int v19; // r12d

  v2 = (_BYTE *)(a1 + 12);
  v3 = 0;
  v4 = 9;
  v5 = unk_4F07580;
  *(_BYTE *)(a1 + 19) = 0;
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
  while ( v3 != 7 );
  if ( unk_4F07580 )
  {
    v11 = a2[7];
    *(_BYTE *)(a1 + 19) = v11;
    v12 = a2[9];
    if ( (v11 & 0x7F) != 0 )
      v7 = 1;
    v13 = (v12 & 0x7F) << 8;
    v14 = 8;
  }
  else
  {
    v11 = a2[2];
    *(_BYTE *)(a1 + 19) = v11;
    v12 = *a2;
    if ( (v11 & 0x7F) != 0 )
      v7 = 1;
    v13 = (v12 & 0x7F) << 8;
    v14 = 1;
  }
  v15 = a2[v14];
  *(_DWORD *)(a1 + 4) = v12 >> 7;
  v16 = v13 | v15;
  if ( v11 < 0 )
  {
    if ( !v16 )
      goto LABEL_15;
    if ( v16 == 0x7FFF )
    {
      if ( v7 )
        *(_DWORD *)a1 = 3;
      else
        *(_DWORD *)a1 = 4;
      result = 16385;
    }
    else
    {
      *(_DWORD *)a1 = 2;
      result = (unsigned int)(v16 - 16382);
    }
  }
  else
  {
    if ( v16 )
    {
LABEL_15:
      *(_DWORD *)a1 = 0;
      result = (unsigned int)(v16 - 16382);
      goto LABEL_16;
    }
    if ( v7 )
    {
      v18 = v2;
      v19 = sub_8EE4D0(v2, 64);
      sub_8EE880(v18, 64, v19);
      *(_DWORD *)a1 = 2;
      result = (unsigned int)(-16381 - v19);
    }
    else
    {
      *(_DWORD *)a1 = 6;
      result = 4294950914LL;
    }
  }
LABEL_16:
  *(_DWORD *)(a1 + 8) = result;
  *(_DWORD *)(a1 + 28) = 64;
  return result;
}
