// Function: sub_2ED4D50
// Address: 0x2ed4d50
//
__int64 __fastcall sub_2ED4D50(unsigned __int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  unsigned __int64 i; // rbx
  __int64 v8; // r13
  _BYTE *v9; // r9
  __int64 result; // rax
  __int64 v11; // r12
  char v12; // al
  __int64 v13; // rcx
  __int64 (*v14)(); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // ecx
  __int64 v18; // rsi
  _BYTE *v19; // r9
  __int64 v20; // rax
  unsigned int v21; // ecx
  __int64 v22; // rsi
  char v23; // al
  __int64 *v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  _BYTE *v26; // [rsp+8h] [rbp-48h]
  _BYTE *v27; // [rsp+10h] [rbp-40h]
  unsigned int v28; // [rsp+10h] [rbp-40h]
  __int64 *v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  for ( i = a1; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v8 = *(_QWORD *)(a1 + 24) + 48LL;
  while ( 1 )
  {
    v9 = *(_BYTE **)(i + 32);
    result = 5LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
    v11 = (__int64)&v9[40 * (*(_DWORD *)(i + 40) & 0xFFFFFF)];
    if ( v9 != (_BYTE *)v11 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v8 == i )
      break;
    if ( (*(_BYTE *)(i + 44) & 4) == 0 )
    {
      i = *(_QWORD *)(a1 + 24) + 48LL;
      break;
    }
  }
  while ( v9 != (_BYTE *)v11 )
  {
    v12 = *v9;
    if ( *v9 == 12 )
      goto LABEL_26;
    while ( 1 )
    {
      if ( !v12 )
      {
        v13 = *((unsigned int *)v9 + 2);
        if ( (unsigned int)(v13 - 1) <= 0x3FFFFFFE )
        {
          if ( (v9[3] & 0x10) != 0 )
          {
            v14 = *(__int64 (**)())(*(_QWORD *)a4 + 168LL);
            if ( v14 == sub_2EA3FB0 )
              goto LABEL_13;
            v24 = a2;
            v26 = v9;
            v28 = *((_DWORD *)v9 + 2);
            v30 = a4;
            v23 = ((__int64 (__fastcall *)(__int64, _QWORD))v14)(a4, (unsigned int)v13);
            a4 = v30;
            v13 = v28;
            v9 = v26;
            a2 = v24;
            if ( !v23 )
            {
LABEL_13:
              v15 = *(_QWORD *)(*a2 + 8);
              v16 = *(_DWORD *)(v15 + 24 * v13 + 16) >> 12;
              v17 = *(_DWORD *)(v15 + 24 * v13 + 16) & 0xFFF;
              v18 = *(_QWORD *)(*a2 + 56) + 2 * v16;
              do
              {
                if ( !v18 )
                  break;
                v18 += 2;
                *(_QWORD *)(a2[1] + 8LL * (v17 >> 6)) |= 1LL << v17;
                v17 += *(__int16 *)(v18 - 2);
              }
              while ( *(_WORD *)(v18 - 2) );
            }
          }
          else
          {
            v20 = *(_DWORD *)(*(_QWORD *)(*a3 + 8LL) + 24 * v13 + 16) >> 12;
            v21 = *(_DWORD *)(*(_QWORD *)(*a3 + 8LL) + 24 * v13 + 16) & 0xFFF;
            v22 = *(_QWORD *)(*a3 + 56LL) + 2 * v20;
            do
            {
              if ( !v22 )
                break;
              v22 += 2;
              *(_QWORD *)(a3[1] + 8LL * (v21 >> 6)) |= 1LL << v21;
              v21 += *(__int16 *)(v22 - 2);
            }
            while ( *(_WORD *)(v22 - 2) );
          }
        }
      }
      v19 = v9 + 40;
      result = v11;
      if ( v19 == (_BYTE *)v11 )
        break;
      v11 = (__int64)v19;
LABEL_25:
      v9 = (_BYTE *)v11;
      v11 = result;
      v12 = *v9;
      if ( *v9 == 12 )
      {
LABEL_26:
        v25 = a4;
        v27 = v9;
        v29 = a2;
        sub_2E21EB0(a2, *((_QWORD *)v9 + 3));
        v9 = v27;
        a4 = v25;
        a2 = v29;
        v12 = *v27;
      }
    }
    while ( 1 )
    {
      i = *(_QWORD *)(i + 8);
      if ( v8 == i )
      {
        v9 = (_BYTE *)v11;
        v11 = result;
        goto LABEL_22;
      }
      if ( (*(_BYTE *)(i + 44) & 4) == 0 )
        break;
      v11 = *(_QWORD *)(i + 32);
      result = v11 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
      if ( v11 != result )
        goto LABEL_25;
    }
    v9 = (_BYTE *)v11;
    i = v8;
    v11 = result;
LABEL_22:
    ;
  }
  return result;
}
