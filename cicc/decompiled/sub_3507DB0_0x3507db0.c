// Function: sub_3507DB0
// Address: 0x3507db0
//
__int64 __fastcall sub_3507DB0(
        _QWORD **a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r8
  unsigned __int64 i; // rbx
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 result; // rax
  unsigned int v13; // esi
  char v14; // al
  __int64 v15; // r14
  unsigned __int16 *v16; // rbx
  __int64 v17; // r12
  unsigned int v18; // esi
  char v19; // al
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  char *v27; // rax
  __int64 v28; // rdx
  char *j; // rdi
  _QWORD *v30; // r11
  __int64 v31; // rsi
  unsigned int v32; // eax
  char *v33; // rdx
  char *v34; // rax
  char *v35; // rax
  const void *v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+18h] [rbp-38h]

  v6 = a3;
  for ( i = a2; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v9 = *(_QWORD *)(a2 + 24) + 48LL;
  while ( 1 )
  {
    v10 = *(_QWORD *)(i + 32);
    v11 = v10 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
    if ( v10 != v11 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v9 == i )
      break;
    if ( (*(_BYTE *)(i + 44) & 4) == 0 )
    {
      i = *(_QWORD *)(a2 + 24) + 48LL;
      break;
    }
  }
  result = a3 + 16;
  v36 = (const void *)(a3 + 16);
  if ( v11 != v10 )
  {
    while ( *(_BYTE *)v10 )
    {
      if ( *(_BYTE *)v10 != 12 )
        goto LABEL_13;
      v23 = v10;
      v15 = v10 + 40;
      v37 = v6;
      sub_3507C70(a1, v23, v6, a4, v6, a6);
      v6 = v37;
      result = v11;
      if ( v15 == v11 )
      {
        while ( 1 )
        {
LABEL_17:
          i = *(_QWORD *)(i + 8);
          if ( v9 == i )
          {
            v10 = v11;
            v11 = result;
            goto LABEL_19;
          }
          if ( (*(_BYTE *)(i + 44) & 4) == 0 )
            break;
          v11 = *(_QWORD *)(i + 32);
          result = v11 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
          if ( v11 != result )
            goto LABEL_33;
        }
        v10 = v11;
        i = v9;
        v11 = result;
LABEL_19:
        if ( v11 == v10 )
          goto LABEL_20;
      }
      else
      {
LABEL_32:
        v11 = v15;
LABEL_33:
        v10 = v11;
        v11 = result;
      }
    }
    if ( (*(_BYTE *)(v10 + 4) & 8) == 0 )
    {
      v13 = *(_DWORD *)(v10 + 8);
      if ( v13 - 1 <= 0x3FFFFFFE )
      {
        v14 = *(_BYTE *)(v10 + 3);
        if ( (v14 & 0x10) != 0 )
        {
          v24 = v40;
          a4 = *(unsigned int *)(v6 + 12);
          LOWORD(v24) = *(_DWORD *)(v10 + 8);
          v40 = v24;
          v25 = *(unsigned int *)(v6 + 8);
          if ( v25 + 1 > a4 )
          {
            v39 = v6;
            sub_C8D5F0(v6, v36, v25 + 1, 0x10u, v6, a6);
            v6 = v39;
            v25 = *(unsigned int *)(v39 + 8);
          }
          v26 = (_QWORD *)(*(_QWORD *)v6 + 16 * v25);
          v26[1] = v10;
          *v26 = v40;
          ++*(_DWORD *)(v6 + 8);
        }
        else if ( (v14 & 0x40) != 0 )
        {
          v38 = v6;
          v27 = sub_E922F0(*a1, v13);
          v6 = v38;
          a6 = (__int64)&v27[2 * v28];
          for ( j = v27; (char *)a6 != j; j += 2 )
          {
            v30 = a1[2];
            v31 = *(unsigned __int16 *)j;
            v32 = *((unsigned __int8 *)a1[6] + v31);
            if ( v32 < (unsigned int)v30 )
            {
              a4 = (unsigned __int64)a1[1];
              while ( 1 )
              {
                v33 = (char *)(a4 + 2LL * v32);
                if ( (_WORD)v31 == *(_WORD *)v33 )
                  break;
                v32 += 256;
                if ( (unsigned int)v30 <= v32 )
                  goto LABEL_47;
              }
              if ( v33 != (char *)(a4 + 2LL * (_QWORD)v30) )
              {
                v34 = (char *)(a4 + 2LL * (_QWORD)v30 - 2);
                if ( v33 != v34 )
                {
                  *(_WORD *)v33 = *(_WORD *)v34;
                  v35 = (char *)a1[1];
                  a4 = *(unsigned __int16 *)&v35[2 * (_QWORD)a1[2] - 2];
                  *((_BYTE *)a1[6] + a4) = (v33 - v35) >> 1;
                  v30 = a1[2];
                }
                a1[2] = (_QWORD *)((char *)v30 - 1);
              }
            }
LABEL_47:
            ;
          }
        }
      }
    }
LABEL_13:
    v15 = v10 + 40;
    result = v11;
    if ( v15 == v11 )
      goto LABEL_17;
    goto LABEL_32;
  }
LABEL_20:
  v16 = *(unsigned __int16 **)v6;
  v17 = *(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8);
  if ( v17 != *(_QWORD *)v6 )
  {
    do
    {
      while ( 1 )
      {
        v21 = (unsigned __int8 *)*((_QWORD *)v16 + 1);
        v22 = *v16;
        v20 = *v21;
        if ( (_BYTE)v20 )
          break;
        v18 = (unsigned __int16)v22;
        v20 = v21[3];
        v19 = (unsigned __int8)v20 >> 4;
        LOBYTE(v20) = (unsigned __int8)v20 >> 6;
        result = v19 & 1;
        if ( ((unsigned __int8)result & (unsigned __int8)v20) == 0 )
          goto LABEL_23;
LABEL_24:
        v16 += 8;
        if ( (unsigned __int16 *)v17 == v16 )
          return result;
      }
      v18 = (unsigned __int16)v22;
      if ( (_BYTE)v20 != 12
        || (v20 = (unsigned __int16)v22 >> 5,
            result = *(unsigned int *)(*((_QWORD *)v21 + 3) + 4 * v20),
            _bittest((const int *)&result, v22)) )
      {
LABEL_23:
        result = sub_3507B80(a1, v18, v20, v22, v6, a6);
        goto LABEL_24;
      }
      v16 += 8;
    }
    while ( (unsigned __int16 *)v17 != v16 );
  }
  return result;
}
