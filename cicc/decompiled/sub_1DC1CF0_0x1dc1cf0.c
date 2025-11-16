// Function: sub_1DC1CF0
// Address: 0x1dc1cf0
//
__int64 __fastcall sub_1DC1CF0(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // rdx
  unsigned __int16 v6; // r12
  unsigned int v7; // edi
  __int16 v8; // r9
  _WORD *v9; // rdi
  unsigned __int16 v10; // si
  unsigned __int16 *v11; // r8
  unsigned __int16 v12; // r9
  unsigned __int16 *v13; // rdi
  unsigned __int16 *v14; // r8
  unsigned __int16 *v15; // rax
  unsigned __int16 v16; // r10
  __int64 result; // rax
  unsigned __int16 *v18; // rcx
  unsigned int v19; // edi
  unsigned int v20; // eax
  __int64 v21; // r12
  _DWORD *v22; // rdx
  __int64 v23; // rax
  _DWORD *v24; // rax
  unsigned __int16 *v25; // rax
  __int64 v26; // rax
  unsigned __int16 v27; // ax

  v2 = *(_QWORD **)a1;
  if ( !*(_QWORD *)a1 )
    BUG();
  v3 = v2[1];
  v4 = v2[7];
  v6 = 0;
  v7 = *(_DWORD *)(v3 + 24LL * a2 + 16);
  v8 = a2 * (v7 & 0xF);
  v9 = (_WORD *)(v4 + 2LL * (v7 >> 4));
  v10 = 0;
  v11 = v9 + 1;
  v12 = *v9 + v8;
LABEL_3:
  v13 = v11;
  while ( 1 )
  {
    v14 = v13;
    if ( !v13 )
    {
      v16 = v6;
      result = 0;
      goto LABEL_7;
    }
    v15 = (unsigned __int16 *)(v2[6] + 4LL * v12);
    v16 = *v15;
    v10 = v15[1];
    if ( *v15 )
      break;
LABEL_25:
    v27 = *v13;
    v11 = 0;
    ++v13;
    if ( !v27 )
      goto LABEL_3;
    v12 += v27;
  }
  while ( 1 )
  {
    result = v4 + 2LL * *(unsigned int *)(v3 + 24LL * v16 + 8);
    if ( result )
      break;
    if ( !v10 )
    {
      v6 = v16;
      goto LABEL_25;
    }
    v16 = v10;
    v10 = 0;
  }
LABEL_7:
  v18 = (unsigned __int16 *)result;
  while ( v14 )
  {
    v19 = *(_DWORD *)(a1 + 16);
    v20 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 56) + v16);
    if ( v20 < v19 )
    {
      v21 = *(_QWORD *)(a1 + 8);
      while ( 1 )
      {
        v22 = (_DWORD *)(v21 + 4LL * v20);
        if ( v16 == *v22 )
          break;
        v20 += 256;
        if ( v19 <= v20 )
          goto LABEL_17;
      }
      v23 = 4LL * v19;
      if ( v22 != (_DWORD *)(v21 + v23) )
      {
        v24 = (_DWORD *)(v21 + v23 - 4);
        if ( v22 != v24 )
        {
          *v22 = *v24;
          *(_BYTE *)(*(_QWORD *)(a1 + 56) + *(unsigned int *)(*(_QWORD *)(a1 + 8) + 4LL * *(unsigned int *)(a1 + 16) - 4)) = ((__int64)v22 - *(_QWORD *)(a1 + 8)) >> 2;
          v19 = *(_DWORD *)(a1 + 16);
        }
        *(_DWORD *)(a1 + 16) = v19 - 1;
      }
    }
LABEL_17:
    result = *v18++;
    v16 += result;
    if ( !(_WORD)result )
    {
      if ( v10 )
      {
        v26 = v10;
        v16 = v10;
        v10 = 0;
        result = v2[7] + 2LL * *(unsigned int *)(v2[1] + 24 * v26 + 8);
      }
      else
      {
        v10 = *v14;
        v12 += *v14;
        if ( *v14 )
        {
          ++v14;
          v25 = (unsigned __int16 *)(v2[6] + 4LL * v12);
          v16 = *v25;
          v10 = v25[1];
          result = v2[7] + 2LL * *(unsigned int *)(v2[1] + 24LL * *v25 + 8);
        }
        else
        {
          result = 0;
          v14 = 0;
        }
      }
      goto LABEL_7;
    }
  }
  return result;
}
