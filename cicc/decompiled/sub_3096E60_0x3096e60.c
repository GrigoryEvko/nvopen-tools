// Function: sub_3096E60
// Address: 0x3096e60
//
__int64 __fastcall sub_3096E60(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  int v7; // eax
  unsigned __int16 v8; // ax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r10
  __int64 v14; // r8
  __int64 v15; // rdx
  _QWORD *v16; // rcx
  __int64 v17; // rdi
  int v18; // r14d
  _QWORD *v19; // rax
  int v20; // edx
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r9
  __int64 v24; // r10
  unsigned int v25; // r14d
  __int64 v26; // r8
  __int64 v27; // rdi
  int v28; // ecx
  _QWORD *v29; // rdx
  int v30; // r8d
  int v31; // r8d
  __int64 v32; // r10
  _QWORD *v33; // rdi
  unsigned int v34; // r9d
  int v35; // ecx
  __int64 v36; // rsi
  int v37; // r8d
  int v38; // r8d
  __int64 v39; // r9
  unsigned int v40; // ecx
  __int64 v41; // rdi
  int v42; // esi
  _QWORD *v43; // r10
  int v44; // r8d
  int v45; // r8d
  __int64 v46; // r9
  unsigned int v47; // r15d
  int v48; // ecx
  __int64 v49; // rsi
  int v50; // r9d
  int v51; // r9d
  __int64 v52; // r10
  unsigned int v53; // ecx
  __int64 v54; // r8
  int v55; // edi
  _QWORD *v56; // rsi
  __int64 v57; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+18h] [rbp-78h]
  int v60; // [rsp+2Ch] [rbp-64h] BYREF
  _BYTE v61[96]; // [rsp+30h] [rbp-60h] BYREF

  result = a2 + 320;
  v57 = a2 + 320;
  v59 = *(_QWORD *)(a2 + 328);
  if ( v59 != a2 + 320 )
  {
    v3 = a1 + 280;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v59 + 56);
      v5 = v59 + 48;
      if ( v4 != v59 + 48 )
        break;
LABEL_25:
      result = *(_QWORD *)(v59 + 8);
      v59 = result;
      if ( v57 == result )
        return result;
    }
    while ( 1 )
    {
      v8 = *(_WORD *)(v4 + 68);
      if ( v8 == 2414 )
        break;
      if ( v8 > 0x96Eu )
      {
        if ( v8 == 2665 )
        {
          v21 = *(_QWORD *)(v4 + 32);
          if ( *(_QWORD *)(v21 + 344) || *(_QWORD *)(v21 + 384) != 8 )
            goto LABEL_12;
        }
        else
        {
          if ( v8 > 0xA69u )
          {
            if ( v8 != 2675 )
              goto LABEL_12;
            v6 = *(_QWORD *)(v4 + 32);
            if ( *(_QWORD *)(v6 + 224) || *(_QWORD *)(v6 + 264) != 8 )
              goto LABEL_12;
            goto LABEL_10;
          }
          if ( v8 != 2420 )
          {
            if ( v8 != 2664 )
              goto LABEL_12;
            v11 = *(_QWORD *)(v4 + 32);
            if ( *(_QWORD *)(v11 + 304) || *(_QWORD *)(v11 + 344) != 8 )
              goto LABEL_12;
            goto LABEL_37;
          }
          v21 = *(_QWORD *)(v4 + 32);
        }
        v60 = *(_DWORD *)(v21 + 8);
        sub_3096C30((__int64)v61, v3, &v60);
        v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 48LL);
        sub_3096C30((__int64)v61, v3, &v60);
        v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 88LL);
        sub_3096C30((__int64)v61, v3, &v60);
        v7 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 128LL);
        goto LABEL_11;
      }
      if ( v8 == 1227 )
      {
        v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 8LL);
        sub_3096C30((__int64)v61, v3, &v60);
        v12 = *(_DWORD *)(a1 + 272);
        v13 = a1 + 248;
        if ( !v12 )
        {
          ++*(_QWORD *)(a1 + 248);
          goto LABEL_70;
        }
        v14 = *(_QWORD *)(a1 + 256);
        LODWORD(v15) = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v16 = (_QWORD *)(v14 + 8LL * (unsigned int)v15);
        v17 = *v16;
        if ( v4 != *v16 )
        {
          v18 = 1;
          v19 = 0;
          while ( v17 != -4096 )
          {
            if ( !v19 && v17 == -8192 )
              v19 = v16;
            v15 = (v12 - 1) & ((_DWORD)v15 + v18);
            v16 = (_QWORD *)(v14 + 8 * v15);
            v17 = *v16;
            if ( v4 == *v16 )
              goto LABEL_12;
            ++v18;
          }
          if ( !v19 )
            v19 = v16;
          ++*(_QWORD *)(a1 + 248);
          v20 = *(_DWORD *)(a1 + 264) + 1;
          if ( 4 * v20 < 3 * v12 )
          {
            if ( v12 - *(_DWORD *)(a1 + 268) - v20 <= v12 >> 3 )
            {
              sub_2E36C70(v13, v12);
              v44 = *(_DWORD *)(a1 + 272);
              if ( !v44 )
              {
LABEL_119:
                ++*(_DWORD *)(a1 + 264);
                BUG();
              }
              v45 = v44 - 1;
              v46 = *(_QWORD *)(a1 + 256);
              v33 = 0;
              v47 = v45 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
              v20 = *(_DWORD *)(a1 + 264) + 1;
              v48 = 1;
              v19 = (_QWORD *)(v46 + 8LL * v47);
              v49 = *v19;
              if ( v4 != *v19 )
              {
                while ( v49 != -4096 )
                {
                  if ( v49 == -8192 && !v33 )
                    v33 = v19;
                  v47 = v45 & (v48 + v47);
                  v19 = (_QWORD *)(v46 + 8LL * v47);
                  v49 = *v19;
                  if ( v4 == *v19 )
                    goto LABEL_46;
                  ++v48;
                }
LABEL_66:
                if ( v33 )
                  v19 = v33;
              }
            }
LABEL_46:
            *(_DWORD *)(a1 + 264) = v20;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a1 + 268);
            *v19 = v4;
            goto LABEL_12;
          }
LABEL_70:
          sub_2E36C70(v13, 2 * v12);
          v37 = *(_DWORD *)(a1 + 272);
          if ( !v37 )
            goto LABEL_119;
          v38 = v37 - 1;
          v39 = *(_QWORD *)(a1 + 256);
          v40 = v38 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v20 = *(_DWORD *)(a1 + 264) + 1;
          v19 = (_QWORD *)(v39 + 8LL * v40);
          v41 = *v19;
          if ( v4 != *v19 )
          {
            v42 = 1;
            v43 = 0;
            while ( v41 != -4096 )
            {
              if ( v41 == -8192 && !v43 )
                v43 = v19;
              v40 = v38 & (v42 + v40);
              v19 = (_QWORD *)(v39 + 8LL * v40);
              v41 = *v19;
              if ( v4 == *v19 )
                goto LABEL_46;
              ++v42;
            }
            if ( v43 )
              v19 = v43;
          }
          goto LABEL_46;
        }
      }
      else
      {
        if ( v8 > 0x4CBu )
        {
          if ( v8 > 0x4CDu && v8 != 2408 )
            goto LABEL_12;
          goto LABEL_9;
        }
        if ( v8 != 325 )
        {
          if ( (unsigned __int16)(v8 - 1220) > 6u )
            goto LABEL_12;
LABEL_9:
          v6 = *(_QWORD *)(v4 + 32);
LABEL_10:
          v7 = *(_DWORD *)(v6 + 8);
LABEL_11:
          v60 = v7;
          sub_3096C30((__int64)v61, v3, &v60);
          goto LABEL_12;
        }
        v9 = *(_QWORD *)(v4 + 32);
        v10 = *(_QWORD *)(v9 + 104);
        if ( v10 != 255 )
        {
          if ( v10 > 0xFE )
            goto LABEL_12;
          v7 = *(_DWORD *)(v9 + 8);
          goto LABEL_11;
        }
        v60 = *(_DWORD *)(v9 + 8);
        sub_3096C30((__int64)v61, v3, &v60);
        v22 = *(_DWORD *)(a1 + 272);
        v23 = a1 + 248;
        if ( !v22 )
        {
          ++*(_QWORD *)(a1 + 248);
          goto LABEL_86;
        }
        v24 = *(_QWORD *)(a1 + 256);
        v25 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
        v26 = (v22 - 1) & v25;
        v19 = (_QWORD *)(v24 + 8 * v26);
        v27 = *v19;
        if ( v4 != *v19 )
        {
          v28 = 1;
          v29 = 0;
          while ( v27 != -4096 )
          {
            if ( !v29 && v27 == -8192 )
              v29 = v19;
            v26 = (v22 - 1) & ((_DWORD)v26 + v28);
            v19 = (_QWORD *)(v24 + 8 * v26);
            v27 = *v19;
            if ( v4 == *v19 )
              goto LABEL_12;
            ++v28;
          }
          if ( v29 )
            v19 = v29;
          ++*(_QWORD *)(a1 + 248);
          v20 = *(_DWORD *)(a1 + 264) + 1;
          if ( 4 * v20 < 3 * v22 )
          {
            if ( v22 - *(_DWORD *)(a1 + 268) - v20 <= v22 >> 3 )
            {
              sub_2E36C70(v23, v22);
              v30 = *(_DWORD *)(a1 + 272);
              if ( !v30 )
                goto LABEL_118;
              v31 = v30 - 1;
              v32 = *(_QWORD *)(a1 + 256);
              v33 = 0;
              v34 = v31 & v25;
              v19 = (_QWORD *)(v32 + 8LL * (v31 & v25));
              v20 = *(_DWORD *)(a1 + 264) + 1;
              v35 = 1;
              v36 = *v19;
              if ( v4 != *v19 )
              {
                while ( v36 != -4096 )
                {
                  if ( !v33 && v36 == -8192 )
                    v33 = v19;
                  v34 = v31 & (v35 + v34);
                  v19 = (_QWORD *)(v32 + 8LL * v34);
                  v36 = *v19;
                  if ( v4 == *v19 )
                    goto LABEL_46;
                  ++v35;
                }
                goto LABEL_66;
              }
            }
            goto LABEL_46;
          }
LABEL_86:
          sub_2E36C70(v23, 2 * v22);
          v50 = *(_DWORD *)(a1 + 272);
          if ( !v50 )
          {
LABEL_118:
            ++*(_DWORD *)(a1 + 264);
            BUG();
          }
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 256);
          v53 = v51 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v20 = *(_DWORD *)(a1 + 264) + 1;
          v19 = (_QWORD *)(v52 + 8LL * v53);
          v54 = *v19;
          if ( v4 != *v19 )
          {
            v55 = 1;
            v56 = 0;
            while ( v54 != -4096 )
            {
              if ( v54 == -8192 && !v56 )
                v56 = v19;
              v53 = v51 & (v55 + v53);
              v19 = (_QWORD *)(v52 + 8LL * v53);
              v54 = *v19;
              if ( v4 == *v19 )
                goto LABEL_46;
              ++v55;
            }
            if ( v56 )
              v19 = v56;
          }
          goto LABEL_46;
        }
      }
LABEL_12:
      if ( (*(_BYTE *)v4 & 4) != 0 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v5 )
          goto LABEL_25;
      }
      else
      {
        while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
          v4 = *(_QWORD *)(v4 + 8);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v5 )
          goto LABEL_25;
      }
    }
    v11 = *(_QWORD *)(v4 + 32);
LABEL_37:
    v60 = *(_DWORD *)(v11 + 8);
    sub_3096C30((__int64)v61, v3, &v60);
    v7 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 48LL);
    goto LABEL_11;
  }
  return result;
}
