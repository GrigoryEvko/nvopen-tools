// Function: sub_AFC700
// Address: 0xafc700
//
__int64 __fastcall sub_AFC700(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 v10; // rcx
  __int64 *v11; // rsi
  __int64 v12; // rsi
  unsigned __int8 v13; // dl
  __int64 v14; // rdi
  unsigned __int8 v15; // dl
  __int64 v16; // rdi
  unsigned __int8 v17; // dl
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rcx
  int v23; // eax
  int v24; // r13d
  unsigned int v25; // eax
  _QWORD *v26; // rcx
  __int64 v27; // rdx
  int v28; // r8d
  _QWORD *v29; // rdi
  __int64 v30; // [rsp+8h] [rbp-48h] BYREF
  __int64 v31; // [rsp+10h] [rbp-40h] BYREF
  __int64 v32; // [rsp+18h] [rbp-38h] BYREF
  __int64 v33; // [rsp+20h] [rbp-30h] BYREF
  __int64 v34[5]; // [rsp+28h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(_BYTE *)(*a2 - 16);
    v10 = *a2 - 16;
    if ( (v9 & 2) != 0 )
      v11 = *(__int64 **)(v6 - 32);
    else
      v11 = (__int64 *)(v10 - 8LL * ((v9 >> 2) & 0xF));
    v12 = *v11;
    v31 = v12;
    v13 = *(_BYTE *)(v6 - 16);
    if ( (v13 & 2) != 0 )
      v14 = *(_QWORD *)(v6 - 32);
    else
      v14 = v10 - 8LL * ((v13 >> 2) & 0xF);
    v32 = *(_QWORD *)(v14 + 8);
    v15 = *(_BYTE *)(v6 - 16);
    if ( (v15 & 2) != 0 )
      v16 = *(_QWORD *)(v6 - 32);
    else
      v16 = v10 - 8LL * ((v15 >> 2) & 0xF);
    v33 = *(_QWORD *)(v16 + 16);
    v17 = *(_BYTE *)(v6 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(v6 - 32);
    else
      v18 = v10 - 8LL * ((v17 >> 2) & 0xF);
    v34[0] = *(_QWORD *)(v18 + 24);
    if ( v12 && *(_BYTE *)v12 == 1 )
    {
      v19 = *(_QWORD *)(v12 + 136);
      v20 = *(__int64 **)(v19 + 24);
      v21 = *(_DWORD *)(v19 + 32);
      if ( v21 > 0x40 )
      {
        v22 = *v20;
      }
      else
      {
        v22 = 0;
        if ( v21 )
          v22 = (__int64)((_QWORD)v20 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
      }
      v30 = v22;
      v23 = sub_AF7D50(&v30, &v32, &v33, v34);
    }
    else
    {
      v23 = sub_AF81D0(&v31, &v32, &v33, v34);
    }
    v24 = v4 - 1;
    v25 = v24 & v23;
    v26 = (_QWORD *)(v7 + 8LL * v25);
    v27 = *v26;
    if ( *v26 == *a2 )
    {
LABEL_31:
      *a3 = v26;
      return 1;
    }
    else
    {
      v28 = 1;
      v29 = 0;
      while ( v27 != -4096 )
      {
        if ( v27 != -8192 || v29 )
          v26 = v29;
        v25 = v24 & (v28 + v25);
        v27 = *(_QWORD *)(v7 + 8LL * v25);
        if ( v27 == *a2 )
        {
          v26 = (_QWORD *)(v7 + 8LL * v25);
          goto LABEL_31;
        }
        ++v28;
        v29 = v26;
        v26 = (_QWORD *)(v7 + 8LL * v25);
      }
      if ( !v29 )
        v29 = v26;
      *a3 = v29;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
