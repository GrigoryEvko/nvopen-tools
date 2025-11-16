// Function: sub_39D3400
// Address: 0x39d3400
//
void __fastcall sub_39D3400(__int64 a1, __int64 a2, char *a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r9
  __int64 *v10; // rax
  char v11; // dl
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  char v14; // al
  __int16 v15; // ax
  __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 *v18; // rsi
  __int64 *v19; // rcx
  const void *v20; // [rsp+0h] [rbp-C0h]
  __int64 v22; // [rsp+18h] [rbp-A8h]
  __int64 v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v25; // [rsp+28h] [rbp-98h]
  __int64 *v26; // [rsp+30h] [rbp-90h]
  __int64 v27; // [rsp+38h] [rbp-88h]
  int v28; // [rsp+40h] [rbp-80h]
  _BYTE v29[120]; // [rsp+48h] [rbp-78h] BYREF

  v4 = a1 + 24;
  v5 = *(_QWORD *)(a1 + 32);
  v25 = (__int64 *)v29;
  v26 = (__int64 *)v29;
  v24 = 0;
  v27 = 8;
  v28 = 0;
  v20 = (const void *)(a2 + 16);
  if ( v5 != a1 + 24 )
  {
    while ( 1 )
    {
      if ( **(_WORD **)(v5 + 16) )
      {
        if ( **(_WORD **)(v5 + 16) != 45 )
        {
          v6 = *(_QWORD *)(v5 + 32);
          v7 = v6 + 40LL * *(unsigned int *)(v5 + 40);
          if ( v6 != v7 )
            break;
        }
      }
LABEL_12:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 46) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        goto LABEL_14;
    }
    v8 = *(_QWORD *)(v5 + 32);
    while ( 1 )
    {
      if ( *(_BYTE *)v8 != 4 )
        goto LABEL_6;
      v9 = *(_QWORD *)(v8 + 24);
      v10 = v25;
      if ( v26 == v25 )
      {
        v18 = &v25[HIDWORD(v27)];
        if ( v25 != v18 )
        {
          v19 = 0;
          while ( v9 != *v10 )
          {
            if ( *v10 == -2 )
              v19 = v10;
            if ( v18 == ++v10 )
            {
              if ( !v19 )
                goto LABEL_33;
              *v19 = v9;
              v12 = *(unsigned int *)(a2 + 8);
              --v28;
              ++v24;
              if ( (unsigned int)v12 < *(_DWORD *)(a2 + 12) )
                goto LABEL_11;
              goto LABEL_32;
            }
          }
          goto LABEL_6;
        }
LABEL_33:
        if ( HIDWORD(v27) < (unsigned int)v27 )
        {
          ++HIDWORD(v27);
          *v18 = v9;
          ++v24;
          goto LABEL_10;
        }
      }
      v22 = *(_QWORD *)(v8 + 24);
      sub_16CCBA0((__int64)&v24, v22);
      v9 = v22;
      if ( v11 )
      {
LABEL_10:
        v12 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 12) )
        {
LABEL_32:
          v23 = v9;
          sub_16CD150(a2, v20, 0, 8, v6, v9);
          v12 = *(unsigned int *)(a2 + 8);
          v9 = v23;
        }
LABEL_11:
        v8 += 40;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v9;
        ++*(_DWORD *)(a2 + 8);
        if ( v7 == v8 )
          goto LABEL_12;
      }
      else
      {
LABEL_6:
        v8 += 40;
        if ( v7 == v8 )
          goto LABEL_12;
      }
    }
  }
LABEL_14:
  v13 = sub_1DD6160(a1);
  v14 = 1;
  if ( v4 != v13 )
  {
    v15 = *(_WORD *)(v13 + 46);
    if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
      v16 = (*(_QWORD *)(*(_QWORD *)(v13 + 16) + 8LL) >> 5) & 1LL;
    else
      LOBYTE(v16) = sub_1E15D00(v13, 0x20u, 1);
    v14 = v16 ^ 1;
  }
  v17 = (unsigned __int64)v26;
  *a3 = v14;
  if ( (__int64 *)v17 != v25 )
    _libc_free(v17);
}
