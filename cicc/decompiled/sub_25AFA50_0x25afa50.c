// Function: sub_25AFA50
// Address: 0x25afa50
//
__int64 __fastcall sub_25AFA50(__int64 a1, unsigned __int64 a2)
{
  unsigned int i; // r12d
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // r13
  unsigned __int8 *v16; // r15
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // r14
  unsigned int *v21; // r12
  __int64 v22; // r13
  __int64 v23; // r15
  unsigned int v24; // ebx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  unsigned int v35; // r14d
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v39; // [rsp+18h] [rbp-B8h]
  unsigned int *v40; // [rsp+20h] [rbp-B0h]
  unsigned int *v41; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-98h]
  _BYTE v43[32]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE v44[32]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+80h] [rbp-50h]

  if ( sub_B2FC80(a2) )
    return 0;
  i = sub_B2FC00((_BYTE *)a2);
  if ( (_BYTE)i )
  {
    return 0;
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 )
      goto LABEL_56;
    v6 = *(_QWORD **)(a1 + 112);
    v7 = (_QWORD *)(a1 + 104);
    if ( v6 )
    {
      v8 = (_QWORD *)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v9 = v6[2];
          v10 = v6[3];
          if ( v6[4] >= a2 )
            break;
          v6 = (_QWORD *)v6[3];
          if ( !v10 )
            goto LABEL_11;
        }
        v8 = v6;
        v6 = (_QWORD *)v6[2];
      }
      while ( v9 );
LABEL_11:
      if ( v7 != v8 && v8[4] <= a2 )
        goto LABEL_56;
    }
    if ( *(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8 )
    {
LABEL_56:
      if ( !(unsigned __int8)sub_B2D610(a2, 20) && *(_QWORD *)(a2 + 16) )
      {
        v41 = (unsigned int *)v43;
        v42 = 0x800000000LL;
        sub_A753E0((__int64)v44);
        if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a2, 20, v11, v12);
          v13 = *(_QWORD *)(a2 + 96);
          v14 = v13 + 40LL * *(_QWORD *)(a2 + 104);
          if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
          {
            sub_B2C6D0(a2, 20, v30, v31);
            v13 = *(_QWORD *)(a2 + 96);
          }
        }
        else
        {
          v13 = *(_QWORD *)(a2 + 96);
          v14 = v13 + 40LL * *(_QWORD *)(a2 + 104);
        }
        for ( i = 0; v14 != v13; v13 += 40 )
        {
          if ( !(unsigned __int8)sub_B2D650(v13) && !*(_QWORD *)(v13 + 16) && !(unsigned __int8)sub_B2BAE0(v13) )
          {
            if ( (*(_BYTE *)(v13 + 7) & 8) != 0 )
            {
              i = 1;
              v37 = sub_ACADE0(*(__int64 ***)(v13 + 8));
              sub_BD84D0(v13, v37);
            }
            v34 = (unsigned int)v42;
            v35 = *(_DWORD *)(v13 + 32);
            v36 = (unsigned int)v42 + 1LL;
            if ( v36 > HIDWORD(v42) )
            {
              sub_C8D5F0((__int64)&v41, v43, v36, 4u, v32, v33);
              v34 = (unsigned int)v42;
            }
            v41[v34] = v35;
            LODWORD(v42) = v42 + 1;
            sub_B2D5D0(a2, *(_DWORD *)(v13 + 32), (__int64)v44);
          }
        }
        if ( (_DWORD)v42 )
        {
          if ( *(_QWORD *)(a2 + 16) )
          {
            v39 = a2;
            v15 = *(_QWORD *)(a2 + 16);
            do
            {
              while ( 1 )
              {
                v16 = *(unsigned __int8 **)(v15 + 24);
                v17 = *v16;
                if ( (unsigned __int8)v17 > 0x1Cu )
                {
                  v18 = (unsigned int)(v17 - 34);
                  if ( (unsigned __int8)v18 <= 0x33u )
                  {
                    v19 = 0x8000000000041LL;
                    if ( _bittest64(&v19, v18) )
                    {
                      if ( (unsigned __int8 *)v15 == v16 - 32 && *((_QWORD *)v16 + 10) == *(_QWORD *)(v39 + 24) )
                      {
                        v40 = &v41[(unsigned int)v42];
                        if ( v41 != v40 )
                          break;
                      }
                    }
                  }
                }
                v15 = *(_QWORD *)(v15 + 8);
                if ( !v15 )
                  goto LABEL_41;
              }
              v38 = v15;
              v20 = (__int64 *)(v16 + 72);
              v21 = v41;
              v22 = *(_QWORD *)(v15 + 24);
              do
              {
                v23 = *v21;
                v24 = *v21;
                v25 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)(v22 + 32 * (v23 - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))) + 8LL));
                v26 = v22 + 32 * (v23 - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF));
                if ( *(_QWORD *)v26 )
                {
                  v27 = *(_QWORD *)(v26 + 8);
                  **(_QWORD **)(v26 + 16) = v27;
                  if ( v27 )
                    *(_QWORD *)(v27 + 16) = *(_QWORD *)(v26 + 16);
                }
                *(_QWORD *)v26 = v25;
                if ( v25 )
                {
                  v28 = *(_QWORD *)(v25 + 16);
                  *(_QWORD *)(v26 + 8) = v28;
                  if ( v28 )
                    *(_QWORD *)(v28 + 16) = v26 + 8;
                  *(_QWORD *)(v26 + 16) = v25 + 16;
                  *(_QWORD *)(v25 + 16) = v26;
                }
                ++v21;
                v29 = (__int64 *)sub_BD5C60(v22);
                *(_QWORD *)(v22 + 72) = sub_A7A440(v20, v29, v24 + 1, (__int64)v44);
              }
              while ( v40 != v21 );
              i = 1;
              v15 = *(_QWORD *)(v38 + 8);
            }
            while ( v15 );
          }
        }
        else
        {
          i = 0;
        }
LABEL_41:
        sub_25AE7E0(v45);
        if ( v41 != (unsigned int *)v43 )
          _libc_free((unsigned __int64)v41);
      }
    }
  }
  return i;
}
