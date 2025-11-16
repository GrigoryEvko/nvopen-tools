// Function: sub_2ECFB30
// Address: 0x2ecfb30
//
_DWORD *__fastcall sub_2ECFB30(__int64 a1, _QWORD *a2)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  void (*v6)(void); // rdx
  void (*v7)(); // rax
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // rdi
  unsigned int v11; // r13d
  int v12; // edx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 *v19; // r15
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int16 *v24; // r15
  char v25; // al
  unsigned int v26; // ecx
  unsigned int v27; // r8d
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  unsigned int *v30; // r15
  unsigned int *v31; // r12
  char v32; // al
  unsigned int v33; // edx
  unsigned int v34; // eax
  unsigned int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rdi
  int v38; // edx
  _DWORD *result; // rax
  __int64 v40; // rax
  __int64 v41; // rsi
  bool v42; // zf
  unsigned int v43; // edi
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // rcx
  __int64 v48; // r12
  __int64 v49; // r8
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // [rsp+8h] [rbp-58h]
  unsigned int v53; // [rsp+10h] [rbp-50h]
  int v54; // [rsp+14h] [rbp-4Ch]
  unsigned __int16 *v55; // [rsp+18h] [rbp-48h]
  unsigned __int16 *v56; // [rsp+18h] [rbp-48h]
  int v57; // [rsp+24h] [rbp-3Ch] BYREF
  unsigned int *v58[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(__int64 **)(a1 + 152);
  if ( *((_DWORD *)v4 + 2) )
  {
    v5 = *v4;
    if ( *(_DWORD *)(a1 + 24) != 1 && (a2[31] & 2) != 0 )
    {
      v6 = *(void (**)(void))(v5 + 32);
      if ( v6 != nullsub_1618 )
      {
        v6();
        v4 = *(__int64 **)(a1 + 152);
        v5 = *v4;
      }
    }
    v7 = *(void (**)())(v5 + 40);
    if ( v7 != nullsub_1619 )
      ((void (__fastcall *)(__int64 *, _QWORD *))v7)(v4, a2);
    *(_BYTE *)(a1 + 160) = 1;
  }
  v8 = a2[2];
  if ( !v8 )
  {
    v48 = *(_QWORD *)a1 + 600LL;
    if ( (unsigned __int8)sub_2FF7B70(v48) )
    {
      v51 = sub_2FF7DB0(v48, *a2);
      a2[2] = v51;
      v8 = v51;
    }
    else
    {
      v8 = a2[2];
    }
  }
  v9 = sub_2FF7F40(*(_QWORD *)(a1 + 8), *a2, 0);
  v10 = *(_QWORD *)(a1 + 8);
  v54 = v9;
  v11 = *(_DWORD *)(a1 + 164);
  v12 = *(_DWORD *)(v10 + 4);
  v13 = *((_DWORD *)a2 + 59);
  if ( *(_DWORD *)(a1 + 24) == 1 )
    v13 = *((_DWORD *)a2 + 58);
  if ( v12 && (v12 == 1 || (*((_BYTE *)a2 + 249) & 0x40) != 0) && v11 < v13 )
    v11 = v13;
  *(_DWORD *)(a1 + 184) += v54;
  if ( (unsigned __int8)sub_2FF7B70(v10) )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) -= *(_DWORD *)(*(_QWORD *)(a1 + 8) + 288LL) * v54;
    v14 = *(unsigned int *)(a1 + 276);
    v15 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v14
      && *(_DWORD *)(v15 + 288) * *(_DWORD *)(a1 + 184) - *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v14) >= *(_DWORD *)(v15 + 292) )
    {
      *(_DWORD *)(a1 + 276) = 0;
    }
    v16 = *(_QWORD *)(v15 + 192);
    v17 = *(unsigned __int16 *)(v8 + 2);
    v18 = *(_QWORD *)(v16 + 176);
    v19 = (unsigned __int16 *)(v18 + 6 * v17);
    v55 = (unsigned __int16 *)(v18 + 6 * (v17 + *(unsigned __int16 *)(v8 + 4)));
    if ( v19 != v55 )
    {
      do
      {
        v20 = sub_2ECFA60(a1, v8, *v19, v19[1], v11, v19[2]);
        if ( v11 < v20 )
          v11 = v20;
        v19 += 3;
      }
      while ( v55 != v19 );
      if ( *((char *)a2 + 249) < 0 )
      {
        v21 = *(_QWORD *)(a1 + 8);
        v22 = *(unsigned __int16 *)(v8 + 2);
        v23 = *(_QWORD *)(*(_QWORD *)(v21 + 192) + 176LL);
        v24 = (unsigned __int16 *)(v23 + 6 * v22);
        v56 = (unsigned __int16 *)(v23 + 6 * (v22 + *(unsigned __int16 *)(v8 + 4)));
        if ( v56 != v24 )
        {
          v52 = a1 + 296;
          while ( 1 )
          {
            if ( *(_DWORD *)(*(_QWORD *)(v21 + 32) + 32LL * *v24 + 16) )
              goto LABEL_26;
            v53 = *v24;
            v25 = sub_2FF85F0();
            v26 = v24[1];
            v27 = v24[2];
            if ( v25 )
              break;
            v28 = sub_2ECE820((_QWORD *)a1, v8, v53, v26, v27);
            v29 = HIDWORD(v28);
            if ( *(_DWORD *)(a1 + 24) == 1 )
            {
              if ( v11 + v24[1] >= (unsigned int)v28 )
                LODWORD(v28) = v11 + v24[1];
              *(_DWORD *)(*(_QWORD *)(a1 + 336) + 4 * v29) = v28;
              goto LABEL_26;
            }
            v24 += 3;
            *(_DWORD *)(*(_QWORD *)(a1 + 336) + 4 * v29) = v11;
            if ( v24 == v56 )
              goto LABEL_32;
LABEL_27:
            v21 = *(_QWORD *)(a1 + 8);
          }
          v40 = sub_2ECE820((_QWORD *)a1, v8, v53, v26, v27);
          v41 = a1 + 296;
          v42 = *(_DWORD *)(a1 + 24) == 1;
          v57 = HIDWORD(v40);
          v43 = HIDWORD(v40);
          v44 = HIDWORD(v40);
          v45 = *(_QWORD *)(a1 + 304);
          if ( v42 )
          {
            if ( !v45 )
              goto LABEL_86;
            do
            {
              while ( 1 )
              {
                v49 = *(_QWORD *)(v45 + 16);
                v50 = *(_QWORD *)(v45 + 24);
                if ( v44 <= *(_DWORD *)(v45 + 32) )
                  break;
                v45 = *(_QWORD *)(v45 + 24);
                if ( !v50 )
                  goto LABEL_84;
              }
              v41 = v45;
              v45 = *(_QWORD *)(v45 + 16);
            }
            while ( v49 );
LABEL_84:
            if ( v41 == v52 || v43 < *(_DWORD *)(v41 + 32) )
            {
LABEL_86:
              v58[0] = (unsigned int *)&v57;
              v41 = sub_2ECE550((_QWORD *)(a1 + 288), v41, v58);
            }
            sub_2ECDCB0(v41 + 40, v11 + (unsigned __int64)v24[2], v11 + (unsigned __int64)v24[1], qword_5021108);
          }
          else
          {
            if ( !v45 )
              goto LABEL_59;
            do
            {
              while ( 1 )
              {
                v46 = *(_QWORD *)(v45 + 16);
                v47 = *(_QWORD *)(v45 + 24);
                if ( v44 <= *(_DWORD *)(v45 + 32) )
                  break;
                v45 = *(_QWORD *)(v45 + 24);
                if ( !v47 )
                  goto LABEL_64;
              }
              v41 = v45;
              v45 = *(_QWORD *)(v45 + 16);
            }
            while ( v46 );
LABEL_64:
            if ( v41 == v52 || v43 < *(_DWORD *)(v41 + 32) )
            {
LABEL_59:
              v58[0] = (unsigned int *)&v57;
              v41 = sub_2ECE550((_QWORD *)(a1 + 288), v41, v58);
            }
            sub_2ECDCB0(v41 + 40, v11 - (unsigned __int64)v24[1] + 1, v11 - (unsigned __int64)v24[2] + 1, qword_5021108);
          }
LABEL_26:
          v24 += 3;
          if ( v24 == v56 )
            goto LABEL_32;
          goto LABEL_27;
        }
      }
    }
  }
LABEL_32:
  v30 = (unsigned int *)(a1 + 180);
  v31 = (unsigned int *)(a1 + 176);
  if ( *(_DWORD *)(a1 + 24) == 1 )
  {
    v31 = (unsigned int *)(a1 + 180);
    v30 = (unsigned int *)(a1 + 176);
  }
  v32 = *((_BYTE *)a2 + 254);
  if ( (v32 & 1) != 0 )
  {
    v33 = *((_DWORD *)a2 + 60);
    if ( v33 <= *v30 )
      goto LABEL_38;
  }
  else
  {
    sub_2F8F5D0(a2);
    v33 = *((_DWORD *)a2 + 60);
    if ( *v30 >= v33 )
      goto LABEL_37;
    if ( (*((_BYTE *)a2 + 254) & 1) == 0 )
    {
      sub_2F8F5D0(a2);
      v33 = *((_DWORD *)a2 + 60);
    }
  }
  *v30 = v33;
LABEL_37:
  v32 = *((_BYTE *)a2 + 254);
LABEL_38:
  if ( (v32 & 2) != 0 )
  {
    v34 = *((_DWORD *)a2 + 61);
    if ( *v31 >= v34 )
      goto LABEL_41;
    goto LABEL_40;
  }
  sub_2F8F770(a2);
  v34 = *((_DWORD *)a2 + 61);
  if ( *v31 < v34 )
  {
    if ( (*((_BYTE *)a2 + 254) & 2) != 0 )
    {
LABEL_40:
      *v31 = v34;
      goto LABEL_41;
    }
    sub_2F8F770(a2);
    *v31 = *((_DWORD *)a2 + 61);
  }
LABEL_41:
  v35 = *(_DWORD *)(a1 + 164);
  if ( v35 < v11 )
  {
    sub_2EC8DA0(a1, v11);
    v37 = *(_QWORD *)(a1 + 8);
  }
  else
  {
    v36 = *(unsigned int *)(a1 + 276);
    v37 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 176) >= v35 )
      v35 = *(_DWORD *)(a1 + 176);
    if ( (_DWORD)v36 )
      v38 = *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v36);
    else
      v38 = *(_DWORD *)(v37 + 288) * *(_DWORD *)(a1 + 184);
    *(_BYTE *)(a1 + 280) = (int)(v38 - *(_DWORD *)(v37 + 292) * v35) >= *(_DWORD *)(v37 + 292);
  }
  *(_DWORD *)(a1 + 168) += v54;
  if ( *(_DWORD *)(a1 + 24) != 1 )
  {
LABEL_48:
    if ( !(unsigned __int8)sub_2FF7E60(v37, *a2, 0) )
      goto LABEL_50;
    goto LABEL_49;
  }
  if ( (unsigned __int8)sub_2FF7ED0(v37, *a2, 0) )
  {
LABEL_49:
    sub_2EC8DA0(a1, ++v11);
    goto LABEL_50;
  }
  if ( *(_DWORD *)(a1 + 24) != 1 )
  {
    v37 = *(_QWORD *)(a1 + 8);
    goto LABEL_48;
  }
LABEL_50:
  result = *(_DWORD **)(a1 + 8);
  if ( *result <= *(_DWORD *)(a1 + 168) )
  {
    do
    {
      sub_2EC8DA0(a1, ++v11);
      result = (_DWORD *)**(unsigned int **)(a1 + 8);
    }
    while ( *(_DWORD *)(a1 + 168) >= (unsigned int)result );
  }
  return result;
}
