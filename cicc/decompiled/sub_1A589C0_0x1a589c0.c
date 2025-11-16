// Function: sub_1A589C0
// Address: 0x1a589c0
//
__int64 __fastcall sub_1A589C0(__int64 a1, __int64 *a2)
{
  char v4; // dl
  __int64 v5; // rdi
  int v6; // esi
  __int64 result; // rax
  __int64 *v8; // r9
  __int64 v9; // r8
  unsigned int v10; // esi
  unsigned int v11; // eax
  __int64 *v12; // r10
  int v13; // ecx
  unsigned int v14; // edi
  int v15; // r11d
  __int64 *v16; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v10 )
    {
      v11 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v12 = 0;
      v13 = (v11 >> 1) + 1;
LABEL_8:
      v14 = 3 * v10;
      goto LABEL_9;
    }
    v6 = v10 - 1;
  }
  result = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (__int64 *)(v5 + 8 * result);
  v9 = *v8;
  if ( *a2 == *v8 )
    return result;
  v15 = 1;
  v12 = 0;
  while ( v9 != -8 )
  {
    if ( v12 || v9 != -16 )
      v8 = v12;
    result = v6 & (unsigned int)(v15 + result);
    v9 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *a2 == v9 )
      return result;
    ++v15;
    v12 = v8;
    v8 = (__int64 *)(v5 + 8LL * (unsigned int)result);
  }
  v11 = *(_DWORD *)(a1 + 8);
  if ( !v12 )
    v12 = v8;
  ++*(_QWORD *)a1;
  v13 = (v11 >> 1) + 1;
  if ( !v4 )
  {
    v10 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v14 = 12;
  v10 = 4;
LABEL_9:
  if ( v14 <= 4 * v13 )
  {
    v10 *= 2;
    goto LABEL_21;
  }
  if ( v10 - *(_DWORD *)(a1 + 12) - v13 <= v10 >> 3 )
  {
LABEL_21:
    sub_1918F70(a1, v10);
    sub_1A54750(a1, a2, &v16);
    v12 = v16;
    v11 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v11 >> 1) + 2) | v11 & 1;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v12 = *a2;
  return sub_15CDD90(a1 + 48, a2);
}
