// Function: sub_2DFE700
// Address: 0x2dfe700
//
__int64 __fastcall sub_2DFE700(__int64 a1, __int64 a2, char a3)
{
  unsigned int v4; // esi
  _BOOL4 v5; // r12d
  __int64 v6; // r14
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  int v12; // r9d
  __int64 i; // rdx
  __int64 v14; // rdi
  __int64 j; // r9
  __int16 v16; // cx
  __int64 v17; // r10
  unsigned int v18; // edi
  unsigned int v19; // esi
  __int64 *v20; // rcx
  __int64 v21; // r9
  _BOOL4 v22; // r13d
  int v23; // edx
  __int64 v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // r13
  unsigned __int64 *v28; // rcx
  unsigned __int64 v29; // rdx
  int v30; // ecx
  int v31; // r8d
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  bool v35; // [rsp+26h] [rbp-3Ah]
  __int64 v37; // [rsp+28h] [rbp-38h]

  v32 = a2 + 320;
  v33 = *(_QWORD *)(a2 + 328);
  if ( v33 == a2 + 320 )
    return 0;
  v4 = 0;
  do
  {
    v5 = v4;
    v6 = *(_QWORD *)(v33 + 56);
    v7 = v33 + 40;
    v37 = v33 + 48;
    if ( v6 == v33 + 48 )
      goto LABEL_31;
    do
    {
      v8 = *(unsigned __int16 *)(v6 + 68);
      v35 = (_WORD)v8 == 24 || (unsigned __int16)(v8 - 14) <= 4u;
      if ( v35 )
      {
        if ( *(_QWORD *)(v33 + 56) == v6 )
        {
          v22 = v5;
          v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 32LL) + 152LL)
                          + 16LL * *(unsigned int *)(v33 + 24));
          goto LABEL_22;
        }
        v9 = *(_QWORD *)(a1 + 112);
        v10 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v10 )
          BUG();
        v11 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
        v12 = *(_DWORD *)(v10 + 44);
        if ( (*(_QWORD *)v10 & 4) != 0 )
        {
          v14 = *(_QWORD *)(v9 + 32);
          if ( (v12 & 4) != 0 )
          {
            do
              v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v11 + 44) & 4) != 0 );
          }
        }
        else
        {
          if ( (v12 & 4) != 0 )
          {
            for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v11 )
            {
              v11 = i & 0xFFFFFFFFFFFFFFF8LL;
              v12 = *(_DWORD *)(v11 + 44) & 0xFFFFFF;
              if ( (*(_DWORD *)(v11 + 44) & 4) == 0 )
                break;
            }
          }
          v14 = *(_QWORD *)(v9 + 32);
          v10 = v11;
        }
        if ( (v12 & 8) != 0 )
        {
          do
            v10 = *(_QWORD *)(v10 + 8);
          while ( (*(_BYTE *)(v10 + 44) & 8) != 0 );
        }
        for ( j = *(_QWORD *)(v10 + 8); j != v11; v11 = *(_QWORD *)(v11 + 8) )
        {
          v16 = *(_WORD *)(v11 + 68);
          if ( (unsigned __int16)(v16 - 14) > 4u && v16 != 24 )
            break;
        }
        v17 = *(_QWORD *)(v14 + 128);
        v18 = *(_DWORD *)(v14 + 144);
        if ( v18 )
        {
          v19 = (v18 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( *v20 == v11 )
          {
LABEL_21:
            v34 = v20[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
            v22 = v5;
LABEL_22:
            while ( a3 )
            {
              v23 = v8 - 16;
              if ( (_WORD)v8 != 14 && (unsigned __int16)(v8 - 16) > 1u )
                break;
              LOBYTE(v23) = (_WORD)v8 == 14 || (unsigned __int16)(v8 - 16) <= 1u;
              v22 = v23;
              v6 = sub_2DF98F0(a1, v6, v34);
LABEL_25:
              if ( v6 != v37 )
              {
                v8 = *(unsigned __int16 *)(v6 + 68);
                if ( (unsigned __int16)(v8 - 14) <= 4u || (_WORD)v8 == 24 )
                  continue;
              }
              v5 = v22;
              goto LABEL_29;
            }
            if ( (unsigned __int16)(v8 - 14) <= 1u )
            {
              if ( (unsigned __int8)sub_2DFD970(a1, v6, v34) )
              {
LABEL_43:
                v25 = v6;
                if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
                {
                  do
                    v25 = *(_QWORD *)(v25 + 8);
                  while ( (*(_BYTE *)(v25 + 44) & 8) != 0 );
                }
                v26 = *(_QWORD *)(v25 + 8);
                while ( v6 != v26 )
                {
                  v27 = (_QWORD *)v6;
                  v6 = *(_QWORD *)(v6 + 8);
                  sub_2E31080(v7, v27);
                  v28 = (unsigned __int64 *)v27[1];
                  v29 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
                  *v28 = v29 | *v28 & 7;
                  *(_QWORD *)(v29 + 8) = v28;
                  *v27 &= 7uLL;
                  v27[1] = 0;
                  sub_2E310F0(v7, v27);
                }
                v22 = v35;
                v6 = v26;
                goto LABEL_25;
              }
              LOWORD(v8) = *(_WORD *)(v6 + 68);
            }
            if ( (_WORD)v8 != 18 || !(unsigned __int8)sub_2DF9A10(a1, v6, v34) )
            {
              if ( (*(_BYTE *)v6 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
                  v6 = *(_QWORD *)(v6 + 8);
              }
              v6 = *(_QWORD *)(v6 + 8);
              goto LABEL_25;
            }
            goto LABEL_43;
          }
          v30 = 1;
          while ( v21 != -4096 )
          {
            v31 = v30 + 1;
            v19 = (v18 - 1) & (v30 + v19);
            v20 = (__int64 *)(v17 + 16LL * v19);
            v21 = *v20;
            if ( *v20 == v11 )
              goto LABEL_21;
            v30 = v31;
          }
        }
        v20 = (__int64 *)(v17 + 16LL * v18);
        goto LABEL_21;
      }
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
      v6 = *(_QWORD *)(v6 + 8);
LABEL_29:
      ;
    }
    while ( v37 != v6 );
    v4 = v5;
LABEL_31:
    v33 = *(_QWORD *)(v33 + 8);
  }
  while ( v32 != v33 );
  return v4;
}
