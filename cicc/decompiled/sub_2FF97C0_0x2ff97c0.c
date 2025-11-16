// Function: sub_2FF97C0
// Address: 0x2ff97c0
//
void __fastcall sub_2FF97C0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  unsigned int *v4; // rax
  unsigned int *v8; // r12
  __int64 v9; // r9
  unsigned int *v10; // rbx
  _QWORD *v11; // r10
  signed int v12; // esi
  int v13; // eax
  _BYTE *v14; // rdi
  _BYTE *v15; // r10
  int v16; // eax
  int v17; // ecx
  __int64 v18; // r8
  int v19; // edx
  unsigned int v20; // eax
  int *v21; // rsi
  int v22; // r9d
  unsigned int v23; // edx
  char v24; // al
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  int v27; // esi
  int v28; // r11d
  _QWORD *v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  unsigned int v31; // [rsp+14h] [rbp-5Ch]
  int v32; // [rsp+14h] [rbp-5Ch]
  __int64 v33; // [rsp+18h] [rbp-58h]
  _QWORD *v34; // [rsp+18h] [rbp-58h]
  _BYTE *v35; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+28h] [rbp-48h]
  _BYTE v37[64]; // [rsp+30h] [rbp-40h] BYREF

  v36 = 0x200000000LL;
  v3 = *(_DWORD *)(a3 + 16);
  v35 = v37;
  if ( v3 )
  {
    v4 = *(unsigned int **)(a3 + 8);
    v8 = &v4[2 * *(unsigned int *)(a3 + 24)];
    if ( v4 != v8 )
    {
      while ( 1 )
      {
        v9 = *v4;
        v10 = v4;
        if ( (unsigned int)v9 <= 0xFFFFFFFD )
          break;
        v4 += 2;
        if ( v8 == v4 )
          return;
      }
      if ( v8 != v4 )
      {
        v11 = &v35;
        while ( 1 )
        {
          v12 = v10[1];
          if ( v12 >= 0 )
          {
            if ( *(_BYTE *)a2 )
            {
              v13 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 4LL * ((unsigned int)v12 >> 5));
              if ( !_bittest(&v13, v12) )
                goto LABEL_29;
            }
            else
            {
              v23 = *(_DWORD *)(a2 + 8);
              if ( v23 == v12
                || (unsigned int)(v12 - 1) <= 0x3FFFFFFE
                && v23 - 1 <= 0x3FFFFFFE
                && (v29 = v11,
                    v31 = v9,
                    v33 = a1,
                    v24 = sub_E92070(*(_QWORD *)(a1 + 16), v12, v23),
                    a1 = v33,
                    v9 = v31,
                    v11 = v29,
                    v24) )
              {
LABEL_29:
                v25 = (unsigned int)v36;
                v26 = (unsigned int)v36 + 1LL;
                if ( v26 > HIDWORD(v36) )
                {
                  v30 = a1;
                  v32 = v9;
                  v34 = v11;
                  sub_C8D5F0((__int64)v11, v37, v26, 4u, a1, v9);
                  v25 = (unsigned int)v36;
                  a1 = v30;
                  LODWORD(v9) = v32;
                  v11 = v34;
                }
                *(_DWORD *)&v35[4 * v25] = v9;
                LODWORD(v36) = v36 + 1;
              }
            }
          }
          v10 += 2;
          if ( v10 == v8 )
            goto LABEL_15;
          while ( *v10 > 0xFFFFFFFD )
          {
            v10 += 2;
            if ( v8 == v10 )
              goto LABEL_15;
          }
          if ( v8 == v10 )
          {
LABEL_15:
            v14 = v35;
            v15 = &v35[4 * (unsigned int)v36];
            if ( v15 != v35 )
            {
              do
              {
                v16 = *(_DWORD *)(a3 + 24);
                v17 = *(_DWORD *)v14;
                v18 = *(_QWORD *)(a3 + 8);
                if ( v16 )
                {
                  v19 = v16 - 1;
                  v20 = (v16 - 1) & (37 * v17);
                  v21 = (int *)(v18 + 8LL * v20);
                  v22 = *v21;
                  if ( *v21 == v17 )
                  {
LABEL_18:
                    *v21 = -2;
                    --*(_DWORD *)(a3 + 16);
                    ++*(_DWORD *)(a3 + 20);
                  }
                  else
                  {
                    v27 = 1;
                    while ( v22 != -1 )
                    {
                      v28 = v27 + 1;
                      v20 = v19 & (v27 + v20);
                      v21 = (int *)(v18 + 8LL * v20);
                      v22 = *v21;
                      if ( *v21 == v17 )
                        goto LABEL_18;
                      v27 = v28;
                    }
                  }
                }
                v14 += 4;
              }
              while ( v15 != v14 );
              v14 = v35;
            }
            if ( v14 != v37 )
              _libc_free((unsigned __int64)v14);
            return;
          }
          v9 = *v10;
        }
      }
    }
  }
}
