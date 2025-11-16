// Function: sub_B3C4F0
// Address: 0xb3c4f0
//
__int64 __fastcall sub_B3C4F0(__int64 a1, __int64 a2, _QWORD *a3)
{
  char *v5; // rdi
  __int64 v6; // rsi
  int v7; // eax
  _QWORD *v8; // r8
  __int64 v9; // r10
  int v10; // r9d
  unsigned int v11; // ecx
  __int64 v12; // r12
  __int16 v13; // r11
  int v14; // eax
  unsigned int v15; // r12d
  size_t v17; // rdx
  int v18; // eax
  _QWORD *v19; // [rsp+8h] [rbp-E8h]
  int v20; // [rsp+10h] [rbp-E0h]
  unsigned int v21; // [rsp+1Ch] [rbp-D4h]
  __int64 v22; // [rsp+20h] [rbp-D0h]
  __int16 v23; // [rsp+28h] [rbp-C8h]
  __int16 v24; // [rsp+42h] [rbp-AEh]
  int v25; // [rsp+44h] [rbp-ACh]
  int v26; // [rsp+54h] [rbp-9Ch]
  __int64 v27; // [rsp+58h] [rbp-98h]
  __int64 v28; // [rsp+70h] [rbp-80h]
  __int16 v29; // [rsp+80h] [rbp-70h]
  void *s1; // [rsp+90h] [rbp-60h]
  __int64 v31; // [rsp+98h] [rbp-58h]
  _QWORD v32[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v33; // [rsp+B0h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v26 = 31;
    v27 = a1 + 16;
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 16);
    v14 = *(_DWORD *)(a1 + 24);
    v26 = v14 - 1;
    if ( !v14 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v5 = *(char **)a2;
  v6 = *(_QWORD *)(a2 + 8);
  v29 = 0;
  LOBYTE(v28) = 0;
  s1 = v32;
  v31 = 0;
  LOBYTE(v32[0]) = 0;
  v33 = 257;
  v7 = sub_B3B940(v5, &v5[v6]);
  v8 = v32;
  v9 = 0;
  v25 = 1;
  v10 = *(unsigned __int16 *)(a2 + 32);
  v11 = v7 & v26;
  v24 = v33;
  while ( 1 )
  {
    v12 = v27 + 48LL * v11;
    v13 = *(_WORD *)(v12 + 32);
    if ( v13 == (_WORD)v10 )
    {
      if ( !*(_BYTE *)(a2 + 32) )
        goto LABEL_15;
      if ( *(_BYTE *)(a2 + 33) )
        goto LABEL_15;
      v17 = *(_QWORD *)(v12 + 8);
      if ( v17 == *(_QWORD *)(a2 + 8) )
      {
        v20 = v10;
        v21 = v11;
        v22 = v9;
        v23 = *(_WORD *)(v12 + 32);
        if ( !v17
          || (v19 = v8,
              v18 = memcmp(*(const void **)v12, *(const void **)a2, v17),
              v8 = v19,
              v13 = v23,
              v9 = v22,
              v11 = v21,
              v10 = v20,
              !v18) )
        {
LABEL_15:
          *a3 = v12;
          v15 = 1;
          goto LABEL_16;
        }
      }
    }
    if ( !v13 )
      break;
    if ( v13 != v24 )
      goto LABEL_10;
    if ( *(_BYTE *)(v12 + 32) && !*(_BYTE *)(v12 + 33) )
      goto LABEL_28;
LABEL_8:
    if ( !v9 )
      v9 = v12;
LABEL_10:
    v11 = v26 & (v25 + v11);
    ++v25;
  }
  if ( *(_BYTE *)(v12 + 32) && !*(_BYTE *)(v12 + 33) && *(_QWORD *)(v12 + 8) )
  {
    if ( v13 != v24 )
      goto LABEL_10;
LABEL_28:
    if ( *(_QWORD *)(v12 + 8) )
      goto LABEL_10;
    goto LABEL_8;
  }
  if ( !v9 )
    v9 = v12;
  v15 = 0;
  *a3 = v9;
LABEL_16:
  if ( v8 != v32 )
    j_j___libc_free_0(v8, v32[0] + 1LL);
  return v15;
}
