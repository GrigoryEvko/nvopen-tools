// Function: sub_2DE6B60
// Address: 0x2de6b60
//
__int64 __fastcall sub_2DE6B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // r15
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // r13d
  const void *v19; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 v21; // [rsp+10h] [rbp-90h]
  __int64 v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  void *dest; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-78h]
  unsigned int v26; // [rsp+2Ch] [rbp-74h]
  _QWORD v27[14]; // [rsp+30h] [rbp-70h] BYREF

  v6 = 1;
  v7 = a1;
  dest = v27;
  v26 = 8;
  v27[0] = a1;
  v19 = (const void *)(a3 + 16);
  while ( 1 )
  {
    v9 = *(unsigned int *)(a3 + 12);
    v25 = v6 - 1;
    v10 = *(unsigned int *)(a3 + 8);
    if ( v10 + 1 > v9 )
    {
      sub_C8D5F0(a3, v19, v10 + 1, 8u, a5, a6);
      v10 = *(unsigned int *)(a3 + 8);
    }
    a6 = 0;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v7;
    ++*(_DWORD *)(a3 + 8);
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 32 * (a6 - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v11 == 85
        && (v14 = *(_QWORD *)(v11 - 32)) != 0
        && !*(_BYTE *)v14
        && *(_QWORD *)(v14 + 24) == *(_QWORD *)(v11 + 80)
        && (*(_BYTE *)(v14 + 33) & 0x20) != 0
        && *(_DWORD *)(v14 + 36) == 383 )
      {
        v15 = v25;
        if ( (unsigned __int64)v25 + 1 > v26 )
        {
          v21 = a6;
          v23 = *(_QWORD *)(v7 + 32 * (a6 - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
          sub_C8D5F0((__int64)&dest, v27, v25 + 1LL, 8u, a5, a6);
          v15 = v25;
          a6 = v21;
          v11 = v23;
        }
        *((_QWORD *)dest + v15) = v11;
        ++v25;
      }
      else
      {
        v12 = *(unsigned int *)(a2 + 8);
        if ( (_DWORD)v12 && *(_QWORD *)(v11 + 8) != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8 * v12 - 8) + 8LL) )
          goto LABEL_26;
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v20 = a6;
          v22 = *(_QWORD *)(v7 + 32 * (a6 - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
          sub_C8D5F0(a2, (const void *)(a2 + 16), v12 + 1, 8u, a5, a6);
          v12 = *(unsigned int *)(a2 + 8);
          a6 = v20;
          v11 = v22;
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v11;
        ++*(_DWORD *)(a2 + 8);
      }
      if ( a6 == 1 )
        break;
      a6 = 1;
    }
    v6 = v25;
    if ( !v25 )
      break;
    v13 = 8LL * v25;
    v7 = *(_QWORD *)dest;
    if ( (char *)dest + 8 != (char *)dest + v13 )
    {
      memmove(dest, (char *)dest + 8, v13 - 8);
      v6 = v25;
    }
  }
  v16 = *(_DWORD *)(a2 + 8);
  if ( v16 <= 1 || (v16 & (v16 - 1)) != 0 )
  {
LABEL_26:
    v17 = 0;
    goto LABEL_27;
  }
  v17 = 1;
  sub_2DE6A20(*(char **)a2, v16);
LABEL_27:
  if ( dest != v27 )
    _libc_free((unsigned __int64)dest);
  return v17;
}
