// Function: sub_1E7A2D0
// Address: 0x1e7a2d0
//
__int64 __fastcall sub_1E7A2D0(unsigned __int64 a1, _QWORD *a2, __int64 *a3, __int64 a4)
{
  unsigned __int64 i; // rbx
  __int64 v8; // r13
  _BYTE *v9; // r9
  __int64 result; // rax
  __int64 v11; // r12
  char v12; // al
  unsigned int v13; // esi
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  _WORD *v16; // rax
  _WORD *v17; // rdi
  unsigned __int64 v18; // rcx
  _WORD *v19; // rsi
  _QWORD *v20; // rax
  int v21; // eax
  _BYTE *v22; // r9
  __int64 v23; // rdi
  _WORD *v24; // rax
  _WORD *v25; // rdi
  unsigned __int64 v26; // rcx
  _WORD *v27; // rsi
  _QWORD *v28; // rax
  int v29; // eax
  char v30; // al
  _QWORD *v31; // [rsp+0h] [rbp-50h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  _BYTE *v33; // [rsp+8h] [rbp-48h]
  _BYTE *v34; // [rsp+10h] [rbp-40h]
  _QWORD *v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  for ( i = a1; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v8 = *(_QWORD *)(a1 + 24) + 24LL;
  do
  {
    v9 = *(_BYTE **)(i + 32);
    result = 5LL * *(unsigned int *)(i + 40);
    v11 = (__int64)&v9[40 * *(unsigned int *)(i + 40)];
    if ( v9 != (_BYTE *)v11 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v8 == i )
      break;
  }
  while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  if ( v9 != (_BYTE *)v11 )
  {
    do
    {
      v12 = *v9;
      if ( *v9 == 12 )
        goto LABEL_26;
      while ( 1 )
      {
        if ( !v12 )
        {
          v13 = *((_DWORD *)v9 + 2);
          if ( (int)v13 > 0 )
          {
            if ( (v9[3] & 0x10) != 0 )
            {
              v14 = *(__int64 (**)())(*(_QWORD *)a4 + 72LL);
              if ( v14 == sub_1E693A0
                || (v31 = a2,
                    v33 = v9,
                    v36 = a4,
                    v30 = ((__int64 (__fastcall *)(__int64))v14)(a4),
                    a4 = v36,
                    v9 = v33,
                    a2 = v31,
                    !v30) )
              {
                v15 = *a2;
                if ( !*a2 )
                  goto LABEL_35;
                v18 = v13 * (*(_DWORD *)(*(_QWORD *)(v15 + 8) + 24LL * v13 + 16) & 0xF);
                v16 = (_WORD *)(*(_QWORD *)(v15 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v15 + 8) + 24LL * v13 + 16) >> 4));
                v17 = v16 + 1;
                LOWORD(v18) = *v16 + v18;
                while ( 1 )
                {
                  v19 = v17;
                  if ( !v17 )
                    break;
                  while ( 1 )
                  {
                    ++v19;
                    v20 = (_QWORD *)(a2[1] + ((v18 >> 3) & 0x1FF8));
                    *v20 |= 1LL << v18;
                    v21 = (unsigned __int16)*(v19 - 1);
                    v17 = 0;
                    if ( !(_WORD)v21 )
                      break;
                    v18 = (unsigned int)(v21 + v18);
                    if ( !v19 )
                      goto LABEL_18;
                  }
                }
              }
            }
            else
            {
              v23 = *a3;
              if ( !*a3 )
LABEL_35:
                BUG();
              v26 = v13 * (*(_DWORD *)(*(_QWORD *)(v23 + 8) + 24LL * v13 + 16) & 0xF);
              v24 = (_WORD *)(*(_QWORD *)(v23 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v23 + 8) + 24LL * v13 + 16) >> 4));
              v25 = v24 + 1;
              LOWORD(v26) = *v24 + v26;
              while ( 1 )
              {
                v27 = v25;
                if ( !v25 )
                  break;
                while ( 1 )
                {
                  ++v27;
                  v28 = (_QWORD *)(a3[1] + ((v26 >> 3) & 0x1FF8));
                  *v28 |= 1LL << v26;
                  v29 = (unsigned __int16)*(v27 - 1);
                  v25 = 0;
                  if ( !(_WORD)v29 )
                    break;
                  v26 = (unsigned int)(v29 + v26);
                  if ( !v27 )
                    goto LABEL_18;
                }
              }
            }
          }
        }
LABEL_18:
        v22 = v9 + 40;
        result = v11;
        if ( v22 == (_BYTE *)v11 )
          break;
        v11 = (__int64)v22;
LABEL_25:
        v9 = (_BYTE *)v11;
        v11 = result;
        v12 = *v9;
        if ( *v9 == 12 )
        {
LABEL_26:
          v32 = a4;
          v34 = v9;
          v35 = a2;
          sub_2103F30(a2, *((_QWORD *)v9 + 3));
          v9 = v34;
          a4 = v32;
          a2 = v35;
          v12 = *v34;
        }
      }
      while ( 1 )
      {
        i = *(_QWORD *)(i + 8);
        if ( v8 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
          break;
        v11 = *(_QWORD *)(i + 32);
        result = v11 + 40LL * *(unsigned int *)(i + 40);
        if ( v11 != result )
          goto LABEL_25;
      }
      v9 = (_BYTE *)v11;
      v11 = result;
    }
    while ( v9 != (_BYTE *)result );
  }
  return result;
}
