// Function: sub_29EC500
// Address: 0x29ec500
//
void __fastcall sub_29EC500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // r15
  __int64 i; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rbx
  _BYTE *v14; // rcx
  __int64 v15; // rax
  _BYTE *v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int8 v20; // dl
  _BYTE **v21; // r8
  __int64 v22; // rcx
  _BYTE **v23; // rbx
  _BYTE **v24; // r14
  __int64 v25; // r8
  __int64 v26; // rax
  _BYTE *v27; // r9
  unsigned __int64 v28; // rdx
  _BYTE *v29; // [rsp+0h] [rbp-E0h]
  __int64 v30; // [rsp+8h] [rbp-D8h]
  _BYTE *v31; // [rsp+18h] [rbp-C8h] BYREF
  _BYTE *v32; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-B8h]
  _BYTE v34[176]; // [rsp+30h] [rbp-B0h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v6 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  v30 = v6;
  if ( v6 != a2 + 72 )
  {
    do
    {
      if ( !v30 )
        BUG();
      v8 = *(_QWORD *)(v30 + 32);
      for ( i = v30 + 24; i != v8; v8 = *(_QWORD *)(v8 + 8) )
      {
        while ( 1 )
        {
          if ( !v8 )
            BUG();
          if ( (*(_BYTE *)(v8 - 17) & 0x20) != 0 )
          {
            v32 = (_BYTE *)sub_B91C10(v8 - 24, 7);
            if ( v32 )
              sub_29EC270(a1, (__int64 *)&v32);
            if ( (*(_BYTE *)(v8 - 17) & 0x20) != 0 )
            {
              v32 = (_BYTE *)sub_B91C10(v8 - 24, 8);
              if ( v32 )
                sub_29EC270(a1, (__int64 *)&v32);
            }
          }
          if ( *(_BYTE *)(v8 - 24) == 85 )
          {
            v10 = *(_QWORD *)(v8 - 56);
            if ( v10 )
            {
              if ( !*(_BYTE *)v10
                && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v8 + 56)
                && (*(_BYTE *)(v10 + 33) & 0x20) != 0
                && *(_DWORD *)(v10 + 36) == 155 )
              {
                break;
              }
            }
          }
          v8 = *(_QWORD *)(v8 + 8);
          if ( i == v8 )
            goto LABEL_20;
        }
        v32 = *(_BYTE **)(*(_QWORD *)(v8 - 24 - 32LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF)) + 24LL);
        sub_29EC270(a1, (__int64 *)&v32);
      }
LABEL_20:
      v30 = *(_QWORD *)(v30 + 8);
    }
    while ( a2 + 72 != v30 );
    v11 = *(unsigned int *)(a1 + 40);
    v12 = *(_QWORD *)(a1 + 32);
    v32 = v34;
    v13 = 8 * v11;
    v33 = 0x1000000000LL;
    if ( v11 > 0x10 )
    {
      sub_C8D5F0((__int64)&v32, v34, v11, 8u, a5, a6);
      v14 = &v32[8 * (unsigned int)v33];
    }
    else
    {
      if ( !v13 )
        return;
      v14 = v34;
    }
    v15 = 0;
    do
    {
      *(_QWORD *)&v14[v15] = *(_QWORD *)(v12 + v15);
      v15 += 8;
    }
    while ( v13 != v15 );
    v16 = v32;
    LODWORD(v33) = v33 + v11;
    v17 = v33;
    while ( v17 )
    {
      v18 = v17--;
      v19 = *(_QWORD *)&v16[8 * v18 - 8];
      LODWORD(v33) = v17;
      v20 = *(_BYTE *)(v19 - 16);
      if ( (v20 & 2) != 0 )
      {
        v21 = *(_BYTE ***)(v19 - 32);
        v22 = *(unsigned int *)(v19 - 24);
      }
      else
      {
        v22 = (*(_WORD *)(v19 - 16) >> 6) & 0xF;
        v21 = (_BYTE **)(v19 + -16 - 8LL * ((v20 >> 2) & 0xF));
      }
      v23 = &v21[v22];
      v24 = v21;
      if ( v23 != v21 )
      {
        do
        {
          while ( 1 )
          {
            if ( (unsigned __int8)(**v24 - 5) <= 0x1Fu )
            {
              v31 = *v24;
              if ( (unsigned __int8)sub_29EC270(a1, (__int64 *)&v31) )
                break;
            }
            if ( v23 == ++v24 )
              goto LABEL_37;
          }
          v26 = (unsigned int)v33;
          v27 = v31;
          v28 = (unsigned int)v33 + 1LL;
          if ( v28 > HIDWORD(v33) )
          {
            v29 = v31;
            sub_C8D5F0((__int64)&v32, v34, v28, 8u, v25, (__int64)v31);
            v26 = (unsigned int)v33;
            v27 = v29;
          }
          ++v24;
          *(_QWORD *)&v32[8 * v26] = v27;
          LODWORD(v33) = v33 + 1;
        }
        while ( v23 != v24 );
LABEL_37:
        v17 = v33;
        v16 = v32;
      }
    }
    if ( v16 != v34 )
      _libc_free((unsigned __int64)v16);
  }
}
