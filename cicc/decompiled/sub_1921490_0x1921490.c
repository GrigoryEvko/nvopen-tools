// Function: sub_1921490
// Address: 0x1921490
//
void __fastcall sub_1921490(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _QWORD *v5; // rax
  unsigned int v6; // eax
  __int64 *v7; // r13
  __int64 v8; // rcx
  int v9; // r8d
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  _DWORD *v13; // r11
  __int64 *v14; // r14
  _DWORD *v15; // rbx
  _DWORD *v16; // r12
  __int64 v17; // rdx
  int v18; // esi
  int v19; // edi
  __int64 v20; // r8
  unsigned __int64 v21; // r9
  unsigned __int64 v22; // r9
  unsigned int i; // eax
  __int64 v24; // r13
  unsigned int v25; // eax
  signed __int64 v26; // rax
  int v27; // edi
  int v28; // r8d
  _DWORD *v29; // rsi
  _DWORD *v30; // rax
  __int64 v31; // rdx
  _DWORD *v32; // r9
  int v33; // eax
  int v36; // [rsp+14h] [rbp-4Ch]
  __int64 v37; // [rsp+18h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 8);
  if ( !v4 )
    return;
  while ( 1 )
  {
    v5 = sub_1648700(v4);
    if ( (unsigned __int8)(*((_BYTE *)v5 + 16) - 25) <= 9u )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      return;
  }
LABEL_10:
  v10 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v10 )
  {
    v9 = 1;
    v11 = *(_QWORD *)(a3 + 8);
    v37 = v5[5];
    v6 = (v10 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
    v7 = (__int64 *)(v11 + 72LL * v6);
    v8 = *v7;
    if ( v37 != *v7 )
    {
      while ( 1 )
      {
        if ( v8 == -8 )
          goto LABEL_8;
        v6 = (v10 - 1) & (v9 + v6);
        v7 = (__int64 *)(v11 + 72LL * v6);
        v8 = *v7;
        if ( v37 == *v7 )
          break;
        ++v9;
      }
    }
    if ( v7 != (__int64 *)(v11 + 72 * v10) )
    {
      v12 = *((unsigned int *)v7 + 4);
      v13 = (_DWORD *)v7[1];
      v14 = v7;
      v15 = v13;
      v16 = &v13[6 * v12];
      if ( v13 != v16 )
      {
        while ( *((_QWORD *)v15 + 1) )
        {
          v15 += 6;
LABEL_16:
          if ( v16 == v15 )
            goto LABEL_8;
        }
        v17 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v17 )
        {
          v18 = *v15;
          v19 = v15[1];
          v36 = 1;
          v20 = *(_QWORD *)(a4 + 8);
          v21 = ((((unsigned int)(37 * v19) | ((unsigned __int64)(unsigned int)(37 * v18) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v19) << 32)) >> 22)
              ^ (((unsigned int)(37 * v19) | ((unsigned __int64)(unsigned int)(37 * v18) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v19) << 32));
          v22 = ((9 * (((v21 - 1 - (v21 << 13)) >> 8) ^ (v21 - 1 - (v21 << 13)))) >> 15)
              ^ (9 * (((v21 - 1 - (v21 << 13)) >> 8) ^ (v21 - 1 - (v21 << 13))));
          for ( i = (v17 - 1) & (((v22 - 1 - (v22 << 27)) >> 31) ^ (v22 - 1 - ((_DWORD)v22 << 27))); ; i = (v17 - 1) & v25 )
          {
            v24 = v20 + 40LL * i;
            if ( *(_DWORD *)v24 == v18 && *(_DWORD *)(v24 + 4) == v19 )
              break;
            if ( *(_DWORD *)v24 == -1 && *(_DWORD *)(v24 + 4) == -1 )
              goto LABEL_27;
            v25 = v36 + i;
            ++v36;
          }
          if ( v24 != v20 + 40 * v17 && *(_DWORD *)(v24 + 16) )
          {
            if ( sub_15CC890(
                   *(_QWORD *)(a1 + 216),
                   v37,
                   *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v24 + 8) + 8LL * *(unsigned int *)(v24 + 16) - 8) + 40LL)) )
            {
              *((_QWORD *)v15 + 1) = a2;
              v31 = *(_QWORD *)(*(_QWORD *)(v24 + 8) + 8LL * (unsigned int)(*(_DWORD *)(v24 + 16))-- - 8);
              *((_QWORD *)v15 + 2) = v31;
            }
            v13 = (_DWORD *)v14[1];
            v12 = *((unsigned int *)v14 + 4);
          }
        }
LABEL_27:
        v26 = 0xAAAAAAAAAAAAAAABLL * (((char *)&v13[6 * v12] - (char *)v15) >> 3);
        if ( v26 >> 2 <= 0 )
        {
          v29 = v15;
          goto LABEL_45;
        }
        v27 = *v15;
        v28 = v15[1];
        v29 = v15;
        v30 = &v15[24 * (v26 >> 2)];
        while ( 1 )
        {
          if ( v29[1] != v28 )
            goto LABEL_30;
          v32 = v29 + 6;
          if ( v27 != v29[6]
            || v28 != v29[7]
            || (v32 = v29 + 12, v27 != v29[12])
            || v28 != v29[13]
            || (v32 = v29 + 18, v27 != v29[18])
            || v28 != v29[19] )
          {
            v29 = v32;
            goto LABEL_30;
          }
          v29 += 24;
          if ( v30 == v29 )
            break;
          if ( v27 != *v29 )
            goto LABEL_30;
        }
        v26 = 0xAAAAAAAAAAAAAAABLL * (((char *)&v13[6 * v12] - (char *)v29) >> 3);
LABEL_45:
        switch ( v26 )
        {
          case 2LL:
            v33 = *v15;
            break;
          case 3LL:
            v33 = *v15;
            if ( *(_QWORD *)v29 != *(_QWORD *)v15 )
              goto LABEL_30;
            v29 += 6;
            break;
          case 1LL:
            v33 = *v15;
LABEL_50:
            if ( v33 == *v29 && v29[1] == v15[1] )
              v29 = &v13[6 * v12];
            goto LABEL_30;
          default:
            v29 = &v13[6 * v12];
LABEL_30:
            v15 = v29;
            goto LABEL_16;
        }
        if ( v33 != *v29 || v29[1] != v15[1] )
          goto LABEL_30;
        v29 += 6;
        goto LABEL_50;
      }
    }
  }
LABEL_8:
  while ( 1 )
  {
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      break;
    v5 = sub_1648700(v4);
    if ( (unsigned __int8)(*((_BYTE *)v5 + 16) - 25) <= 9u )
      goto LABEL_10;
  }
}
