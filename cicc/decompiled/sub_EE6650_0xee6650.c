// Function: sub_EE6650
// Address: 0xee6650
//
__int64 *__fastcall sub_EE6650(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // cl
  int v5; // ecx
  __int64 v6; // rdi
  int v7; // r8d
  unsigned int v8; // edx
  __int64 *result; // rax
  __int64 v10; // r9
  unsigned int v11; // r8d
  unsigned int v12; // eax
  int v13; // edx
  unsigned int v14; // edi
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // esi
  __int64 *v18; // [rsp+8h] [rbp-28h] BYREF
  __int64 v19; // [rsp+10h] [rbp-20h] BYREF
  __int64 v20; // [rsp+18h] [rbp-18h]

  v4 = *(_BYTE *)(a1 + 144);
  v19 = a2;
  v20 = a3;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 152;
    v7 = 31;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 160);
    v6 = *(_QWORD *)(a1 + 152);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 144);
      ++*(_QWORD *)(a1 + 136);
      v18 = 0;
      v13 = (v12 >> 1) + 1;
LABEL_8:
      v14 = 3 * v11;
      goto LABEL_9;
    }
    v7 = v11 - 1;
  }
  v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v6 + 16LL * v8);
  v10 = *result;
  if ( a2 == *result )
    return result;
  v15 = 1;
  v16 = 0;
  while ( v10 != -4096 )
  {
    if ( v16 || v10 != -8192 )
      result = v16;
    v8 = v7 & (v15 + v8);
    v10 = *(_QWORD *)(v6 + 16LL * v8);
    if ( a2 == v10 )
      return result;
    ++v15;
    v16 = result;
    result = (__int64 *)(v6 + 16LL * v8);
  }
  if ( !v16 )
    v16 = result;
  v12 = *(_DWORD *)(a1 + 144);
  ++*(_QWORD *)(a1 + 136);
  v18 = v16;
  v13 = (v12 >> 1) + 1;
  if ( !(_BYTE)v5 )
  {
    v11 = *(_DWORD *)(a1 + 160);
    goto LABEL_8;
  }
  v14 = 96;
  v11 = 32;
LABEL_9:
  if ( v14 <= 4 * v13 )
  {
    v17 = 2 * v11;
LABEL_21:
    sub_EE6200(a1 + 136, v17);
    sub_EE6130(a1 + 136, &v19, &v18);
    a2 = v19;
    v12 = *(_DWORD *)(a1 + 144);
    goto LABEL_11;
  }
  if ( v11 - *(_DWORD *)(a1 + 148) - v13 <= v11 >> 3 )
  {
    v17 = v11;
    goto LABEL_21;
  }
LABEL_11:
  *(_DWORD *)(a1 + 144) = (2 * (v12 >> 1) + 2) | v12 & 1;
  result = v18;
  if ( *v18 != -4096 )
    --*(_DWORD *)(a1 + 148);
  *result = a2;
  result[1] = v20;
  return result;
}
