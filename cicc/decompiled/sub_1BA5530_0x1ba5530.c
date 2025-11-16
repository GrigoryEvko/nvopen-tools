// Function: sub_1BA5530
// Address: 0x1ba5530
//
__int64 __fastcall sub_1BA5530(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // r15
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r12
  _QWORD *v10; // r13
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // r12
  unsigned int v14; // esi
  unsigned int v15; // r8d
  __int64 v16; // rcx
  __int64 v17; // r9
  unsigned __int64 v18; // r13
  char v19; // al
  __int64 v20; // rsi
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rax
  int v27; // r11d
  __int64 v28; // rsi
  int v29; // r11d
  unsigned int v30; // eax
  __int64 v31; // rcx
  int v32; // edx
  unsigned __int8 v33; // al
  _QWORD *v34; // rcx
  int v35; // edi
  __int64 v36; // r10
  unsigned int v37; // r11d
  int v38; // eax
  __int64 v39; // rdi
  __int64 *v40; // r10
  __int64 v41; // rax
  int v42; // r11d
  int v43; // eax
  int v44; // eax
  __int64 v45; // r10
  __int64 v46; // [rsp+8h] [rbp-128h]
  __int64 v47; // [rsp+10h] [rbp-120h]
  __int64 v48; // [rsp+18h] [rbp-118h]
  __int64 v49; // [rsp+20h] [rbp-110h]
  __int64 v50; // [rsp+28h] [rbp-108h]
  unsigned int v51; // [rsp+34h] [rbp-FCh]
  __int64 v52; // [rsp+38h] [rbp-F8h]
  __int64 v53; // [rsp+48h] [rbp-E8h] BYREF
  unsigned __int64 v54[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+60h] [rbp-D0h]
  __int64 v56; // [rsp+68h] [rbp-C8h]
  __int64 v57; // [rsp+70h] [rbp-C0h]
  __int64 v58; // [rsp+78h] [rbp-B8h]
  __int64 v59; // [rsp+80h] [rbp-B0h]
  char v60; // [rsp+88h] [rbp-A8h]
  _QWORD v61[2]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v62; // [rsp+A0h] [rbp-90h]
  char v63[120]; // [rsp+B8h] [rbp-78h] BYREF

  v1 = a1;
  v2 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 368) + 40LL));
  v3 = *(_QWORD *)(a1 + 296);
  v49 = v2;
  v46 = *(_QWORD *)(v3 + 40);
  if ( v46 == *(_QWORD *)(v3 + 32) )
  {
    LODWORD(v50) = 8;
    v51 = -1;
    return (v50 << 32) | v51;
  }
  v47 = *(_QWORD *)(v3 + 32);
  LODWORD(v50) = 8;
  v51 = -1;
  v48 = a1 + 392;
  do
  {
    v52 = *(_QWORD *)v47 + 40LL;
    if ( *(_QWORD *)(*(_QWORD *)v47 + 48LL) == v52 )
      goto LABEL_38;
    v4 = *(_QWORD *)(*(_QWORD *)v47 + 48LL);
    v5 = v1;
    do
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        v6 = *(_QWORD **)(v5 + 408);
        v7 = *(_QWORD **)(v5 + 400);
        v8 = v4 - 24;
        v9 = *(_QWORD *)(v4 - 24);
        if ( v6 == v7 )
        {
          v10 = &v7[*(unsigned int *)(v5 + 420)];
          if ( v7 == v10 )
          {
            v34 = *(_QWORD **)(v5 + 400);
          }
          else
          {
            do
            {
              if ( v8 == *v7 )
                break;
              ++v7;
            }
            while ( v10 != v7 );
            v34 = v10;
          }
        }
        else
        {
          v10 = &v6[*(unsigned int *)(v5 + 416)];
          v7 = sub_16CC9F0(v48, v4 - 24);
          if ( v8 == *v7 )
          {
            v24 = *(_QWORD *)(v5 + 408);
            v25 = v24 == *(_QWORD *)(v5 + 400) ? *(unsigned int *)(v5 + 420) : *(unsigned int *)(v5 + 416);
            v34 = (_QWORD *)(v24 + 8 * v25);
          }
          else
          {
            v11 = *(_QWORD *)(v5 + 408);
            if ( v11 != *(_QWORD *)(v5 + 400) )
            {
              v7 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v5 + 416));
              goto LABEL_11;
            }
            v7 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v5 + 420));
            v34 = v7;
          }
        }
        for ( ; v34 != v7; ++v7 )
        {
          if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
LABEL_11:
        if ( v7 != v10 )
          goto LABEL_5;
        v12 = *(_BYTE *)(v4 - 8);
        if ( v12 != 54 && v12 != 55 )
        {
          if ( v12 != 77 )
            goto LABEL_5;
          v13 = *(_QWORD *)(v5 + 320);
          v53 = v4 - 24;
          v14 = *(_DWORD *)(v13 + 96);
          if ( !v14 )
            goto LABEL_5;
          v15 = v14 - 1;
          v16 = *(_QWORD *)(v13 + 80);
          LODWORD(v17) = (v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v18 = v16 + 176LL * (unsigned int)v17;
          if ( v8 != *(_QWORD *)v18 )
          {
            v35 = 1;
            v36 = *(_QWORD *)(v16 + 176LL * ((v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
            v37 = (v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            while ( 1 )
            {
              if ( v36 == -8 )
                goto LABEL_5;
              v38 = v35 + 1;
              v39 = v15 & (v37 + v35);
              v37 = v39;
              v36 = *(_QWORD *)(v16 + 176 * v39);
              if ( v8 == v36 )
                break;
              v35 = v38;
            }
            v40 = (__int64 *)(v16 + 176LL * ((v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
            v41 = *v40;
            v42 = 1;
            v18 = 0;
            while ( v41 != -8 )
            {
              if ( !v18 && v41 == -16 )
                v18 = (unsigned __int64)v40;
              v17 = v15 & ((_DWORD)v17 + v42);
              v40 = (__int64 *)(v16 + 176 * v17);
              v41 = *v40;
              if ( v8 == *v40 )
              {
                v18 = v16 + 176 * v17;
                goto LABEL_17;
              }
              ++v42;
            }
            v43 = *(_DWORD *)(v13 + 88);
            if ( !v18 )
              v18 = (unsigned __int64)v40;
            ++*(_QWORD *)(v13 + 72);
            v44 = v43 + 1;
            if ( 4 * v44 >= 3 * v14 )
            {
              v14 *= 2;
            }
            else
            {
              v45 = v4 - 24;
              if ( v14 - *(_DWORD *)(v13 + 92) - v44 > v14 >> 3 )
              {
LABEL_74:
                *(_DWORD *)(v13 + 88) = v44;
                if ( *(_QWORD *)v18 != -8 )
                  --*(_DWORD *)(v13 + 92);
                *(_QWORD *)v18 = v45;
                memset((void *)(v18 + 8), 0, 0xA8u);
                *(_QWORD *)(v18 + 8) = 6;
                *(_QWORD *)(v18 + 80) = v18 + 112;
                *(_QWORD *)(v18 + 88) = v18 + 112;
                *(_QWORD *)(v18 + 96) = 8;
                goto LABEL_17;
              }
            }
            sub_1BA42A0(v13 + 72, v14);
            sub_1BA0DE0(v13 + 72, &v53, v54);
            v18 = v54[0];
            v45 = v53;
            v44 = *(_DWORD *)(v13 + 88) + 1;
            goto LABEL_74;
          }
LABEL_17:
          v54[0] = 6;
          v54[1] = 0;
          v55 = *(_QWORD *)(v18 + 24);
          if ( v55 != -8 && v55 != 0 && v55 != -16 )
            sub_1649AC0(v54, *(_QWORD *)(v18 + 8) & 0xFFFFFFFFFFFFFFF8LL);
          v56 = *(_QWORD *)(v18 + 32);
          v57 = *(_QWORD *)(v18 + 40);
          v58 = *(_QWORD *)(v18 + 48);
          v59 = *(_QWORD *)(v18 + 56);
          v60 = *(_BYTE *)(v18 + 64);
          sub_16CCCB0(v61, (__int64)v63, v18 + 72);
          v9 = v59;
          if ( v62 != v61[1] )
            _libc_free(v62);
          if ( v55 != -8 && v55 != 0 && v55 != -16 )
            sub_1649B30(v54);
        }
        if ( *(_BYTE *)(v4 - 8) == 55 )
          v9 = **(_QWORD **)(v4 - 72);
        v19 = *(_BYTE *)(v9 + 8);
        if ( v19 != 15 )
          break;
        if ( sub_1B92F00(v5, v4 - 24) )
          goto LABEL_55;
        v26 = *(_QWORD *)(v5 + 384);
        v27 = *(_DWORD *)(v26 + 72);
        if ( v27 )
        {
          v28 = *(_QWORD *)(v26 + 56);
          v29 = v27 - 1;
          v30 = v29 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v31 = *(_QWORD *)(v28 + 16LL * v30);
          if ( v8 == v31 )
            goto LABEL_55;
          v32 = 1;
          while ( v31 != -8 )
          {
            v30 = v29 & (v32 + v30);
            v31 = *(_QWORD *)(v28 + 16LL * v30);
            if ( v8 == v31 )
              goto LABEL_55;
            ++v32;
          }
        }
        v33 = *(_BYTE *)(v4 - 8);
        if ( v33 > 0x17u )
        {
          if ( v33 == 55 )
          {
            if ( (unsigned __int8)sub_14A2BC0(*(_QWORD *)(v5 + 328)) )
              goto LABEL_55;
          }
          else if ( v33 == 54 && (unsigned __int8)sub_14A2B90(*(_QWORD *)(v5 + 328)) )
          {
LABEL_55:
            v19 = *(_BYTE *)(v9 + 8);
            break;
          }
        }
LABEL_5:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v52 == v4 )
          goto LABEL_37;
      }
      v20 = v9;
      if ( v19 == 16 )
        v20 = **(_QWORD **)(v9 + 16);
      v21 = sub_127FA20(v49, v20);
      if ( v51 <= v21 )
        v21 = v51;
      v51 = v21;
      if ( *(_BYTE *)(v9 + 8) == 16 )
        v9 = **(_QWORD **)(v9 + 16);
      v22 = sub_127FA20(v49, v9);
      v4 = *(_QWORD *)(v4 + 8);
      if ( (unsigned int)v50 >= v22 )
        v22 = v50;
      LODWORD(v50) = v22;
    }
    while ( v52 != v4 );
LABEL_37:
    v1 = v5;
LABEL_38:
    v47 += 8;
  }
  while ( v46 != v47 );
  return (v50 << 32) | v51;
}
