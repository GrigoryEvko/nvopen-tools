// Function: sub_38663E0
// Address: 0x38663e0
//
void __fastcall sub_38663E0(__int64 a1, __int64 a2, char a3, __int64 a4, int a5, int a6)
{
  int v7; // r8d
  int v8; // r9d
  _BYTE *v9; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // r13d
  _BYTE *v14; // rsi
  _BYTE *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rsi
  _BYTE *v25; // [rsp+0h] [rbp-70h] BYREF
  __int64 v26; // [rsp+8h] [rbp-68h]
  _BYTE src[96]; // [rsp+10h] [rbp-60h] BYREF

  sub_38653C0(a1, a2, a3, a4, a5, a6);
  sub_385DCB0((__int64)&v25, a1);
  v9 = v25;
  if ( v25 != src )
  {
    v10 = *(_QWORD *)(a1 + 272);
    if ( v10 != a1 + 288 )
    {
      _libc_free(v10);
      v9 = v25;
    }
    *(_QWORD *)(a1 + 272) = v9;
    *(_QWORD *)(a1 + 280) = v26;
    return;
  }
  v11 = (unsigned int)v26;
  v12 = *(unsigned int *)(a1 + 280);
  v13 = v26;
  if ( (unsigned int)v26 > v12 )
  {
    if ( (unsigned int)v26 > (unsigned __int64)*(unsigned int *)(a1 + 284) )
    {
      *(_DWORD *)(a1 + 280) = 0;
      sub_16CD150(a1 + 272, (const void *)(a1 + 288), v11, 16, v7, v8);
      v15 = v25;
      v11 = (unsigned int)v26;
      v12 = 0;
      v14 = v25;
    }
    else
    {
      v14 = src;
      v15 = src;
      if ( *(_DWORD *)(a1 + 280) )
      {
        v17 = *(_QWORD *)(a1 + 272);
        v12 *= 16LL;
        v18 = (__int64 *)src;
        v19 = v17 + v12;
        do
        {
          v20 = *v18;
          v17 += 16;
          v18 += 2;
          *(_QWORD *)(v17 - 16) = v20;
          *(_QWORD *)(v17 - 8) = *(v18 - 1);
        }
        while ( v17 != v19 );
        v15 = v25;
        v11 = (unsigned int)v26;
        v14 = &v25[v12];
      }
    }
    v16 = 16 * v11;
    if ( v14 == &v15[v16] )
      goto LABEL_12;
    memcpy((void *)(v12 + *(_QWORD *)(a1 + 272)), v14, v16 - v12);
    goto LABEL_11;
  }
  v15 = src;
  if ( (_DWORD)v26 )
  {
    v21 = *(_QWORD *)(a1 + 272);
    v22 = (__int64 *)src;
    v23 = v21 + 16LL * (unsigned int)v26;
    do
    {
      v24 = *v22;
      v21 += 16;
      v22 += 2;
      *(_QWORD *)(v21 - 16) = v24;
      *(_QWORD *)(v21 - 8) = *(v22 - 1);
    }
    while ( v21 != v23 );
LABEL_11:
    v15 = v25;
  }
LABEL_12:
  *(_DWORD *)(a1 + 280) = v13;
  if ( v15 != src )
    _libc_free((unsigned __int64)v15);
}
