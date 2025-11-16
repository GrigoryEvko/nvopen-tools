// Function: sub_38570B0
// Address: 0x38570b0
//
void __fastcall sub_38570B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // r15d
  int v6; // edx
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rbx
  int v13; // eax
  __int64 i; // r12
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // r10
  int v18; // r12d
  __int64 *v19; // rbx
  unsigned int v20; // ecx
  __int64 *v21; // r9
  __int64 v22; // r8
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rdx
  int v26; // eax
  int v27; // esi
  __int64 v28; // r8
  int v29; // ecx
  unsigned int v30; // eax
  int v31; // r10d
  __int64 *v32; // r9
  int v33; // eax
  _BYTE *v34; // rsi
  __int64 v35; // r13
  unsigned __int64 v36; // rdi
  unsigned int v37; // ebx
  unsigned __int64 v38; // r13
  int v39; // edx
  int v40; // edi
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // r15
  int v45; // eax
  __int64 j; // r12
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rax
  int v50; // eax
  int v51; // esi
  int v52; // r10d
  __int64 *v53; // r9
  __int64 v54; // r8
  unsigned int v55; // eax
  __int64 v56; // rdi
  __int64 v57; // [rsp+8h] [rbp-C8h]
  unsigned int v58; // [rsp+14h] [rbp-BCh]
  unsigned __int64 v59; // [rsp+38h] [rbp-98h]
  int v61; // [rsp+48h] [rbp-88h]
  int v62; // [rsp+4Ch] [rbp-84h]
  __int64 v63; // [rsp+58h] [rbp-78h] BYREF
  __int64 v64; // [rsp+60h] [rbp-70h] BYREF
  __int64 v65; // [rsp+68h] [rbp-68h] BYREF
  _QWORD *v66; // [rsp+70h] [rbp-60h] BYREF
  __int64 v67; // [rsp+78h] [rbp-58h]
  _QWORD v68[10]; // [rsp+80h] [rbp-50h] BYREF

  v63 = a1;
  v4 = sub_157EBA0(a2);
  if ( v4 )
  {
    v61 = sub_15F4D60(v4);
    v59 = sub_157EBA0(a2);
    if ( v61 )
    {
      v5 = 0;
      while ( 1 )
      {
        v11 = sub_15F4DF0(v59, v5);
        v12 = v11;
        if ( a3 == v11 )
          goto LABEL_5;
        v66 = (_QWORD *)v11;
        v13 = *(_DWORD *)(a1 + 288);
        if ( v13 )
        {
          v6 = v13 - 1;
          v7 = *(_QWORD *)(a1 + 272);
          v8 = 1;
          v9 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v10 = *(_QWORD *)(v7 + 8LL * v9);
          if ( v12 != v10 )
          {
            while ( v10 != -8 )
            {
              v9 = v6 & (v8 + v9);
              v10 = *(_QWORD *)(v7 + 8LL * v9);
              if ( v12 == v10 )
                goto LABEL_5;
              ++v8;
            }
            goto LABEL_8;
          }
LABEL_5:
          if ( v61 == ++v5 )
            return;
        }
        else
        {
LABEL_8:
          for ( i = *(_QWORD *)(v12 + 8); i; i = *(_QWORD *)(i + 8) )
          {
            if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
              break;
          }
          if ( !sub_38520A0(i, 0, &v63, (__int64 *)&v66) )
            goto LABEL_5;
          v15 = v68;
          v68[0] = v12;
          v67 = 0x400000001LL;
          v57 = a1 + 264;
          v16 = 1;
          v66 = v68;
          v58 = v5;
          do
          {
            while ( 1 )
            {
              v23 = v16;
              v24 = *(_DWORD *)(a1 + 288);
              --v16;
              v25 = v15[v23 - 1];
              LODWORD(v67) = v16;
              v64 = v25;
              if ( !v24 )
              {
                ++*(_QWORD *)(a1 + 264);
LABEL_16:
                sub_13B3D40(v57, 2 * v24);
                v26 = *(_DWORD *)(a1 + 288);
                if ( !v26 )
                  goto LABEL_84;
                v27 = v26 - 1;
                v28 = *(_QWORD *)(a1 + 272);
                v29 = *(_DWORD *)(a1 + 280) + 1;
                v30 = (v26 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                v19 = (__int64 *)(v28 + 8LL * v30);
                v25 = *v19;
                if ( v64 != *v19 )
                {
                  v31 = 1;
                  v32 = 0;
                  while ( v25 != -8 )
                  {
                    if ( v25 == -16 && !v32 )
                      v32 = v19;
                    v30 = v27 & (v31 + v30);
                    v19 = (__int64 *)(v28 + 8LL * v30);
                    v25 = *v19;
                    if ( v64 == *v19 )
                      goto LABEL_32;
                    ++v31;
                  }
                  v25 = v64;
                  if ( v32 )
                    v19 = v32;
                }
                goto LABEL_32;
              }
              v17 = *(_QWORD *)(a1 + 272);
              v18 = 1;
              v19 = 0;
              v20 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v21 = (__int64 *)(v17 + 8LL * v20);
              v22 = *v21;
              if ( v25 != *v21 )
                break;
LABEL_13:
              if ( !v16 )
                goto LABEL_51;
            }
            while ( v22 != -8 )
            {
              if ( v22 != -16 || v19 )
                v21 = v19;
              v20 = (v24 - 1) & (v18 + v20);
              v22 = *(_QWORD *)(v17 + 8LL * v20);
              if ( v25 == v22 )
                goto LABEL_13;
              ++v18;
              v19 = v21;
              v21 = (__int64 *)(v17 + 8LL * v20);
            }
            v33 = *(_DWORD *)(a1 + 280);
            if ( !v19 )
              v19 = v21;
            ++*(_QWORD *)(a1 + 264);
            v29 = v33 + 1;
            if ( 4 * (v33 + 1) >= 3 * v24 )
              goto LABEL_16;
            if ( v24 - *(_DWORD *)(a1 + 284) - v29 <= v24 >> 3 )
            {
              sub_13B3D40(v57, v24);
              v50 = *(_DWORD *)(a1 + 288);
              if ( !v50 )
              {
LABEL_84:
                ++*(_DWORD *)(a1 + 280);
                BUG();
              }
              v25 = v64;
              v51 = v50 - 1;
              v52 = 1;
              v53 = 0;
              v54 = *(_QWORD *)(a1 + 272);
              v55 = (v50 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v19 = (__int64 *)(v54 + 8LL * v55);
              v56 = *v19;
              v29 = *(_DWORD *)(a1 + 280) + 1;
              if ( *v19 != v64 )
              {
                while ( v56 != -8 )
                {
                  if ( v56 == -16 && !v53 )
                    v53 = v19;
                  v55 = v51 & (v52 + v55);
                  v19 = (__int64 *)(v54 + 8LL * v55);
                  v56 = *v19;
                  if ( v64 == *v19 )
                    goto LABEL_32;
                  ++v52;
                }
                if ( v53 )
                  v19 = v53;
              }
            }
LABEL_32:
            *(_DWORD *)(a1 + 280) = v29;
            if ( *v19 != -8 )
              --*(_DWORD *)(a1 + 284);
            *v19 = v25;
            v34 = *(_BYTE **)(a1 + 304);
            if ( v34 == *(_BYTE **)(a1 + 312) )
            {
              sub_1292090(a1 + 296, v34, &v64);
              v35 = v64;
            }
            else
            {
              v35 = v64;
              if ( v34 )
              {
                *(_QWORD *)v34 = v64;
                v34 = *(_BYTE **)(a1 + 304);
                v35 = v64;
              }
              *(_QWORD *)(a1 + 304) = v34 + 8;
            }
            v36 = sub_157EBA0(v35);
            if ( v36 )
            {
              v37 = 0;
              v62 = sub_15F4D60(v36);
              v38 = sub_157EBA0(v35);
              if ( v62 )
              {
                while ( 1 )
                {
                  v65 = sub_15F4DF0(v38, v37);
                  v44 = v65;
                  v45 = *(_DWORD *)(a1 + 288);
                  if ( !v45 )
                    goto LABEL_44;
                  v39 = v45 - 1;
                  v40 = 1;
                  v41 = *(_QWORD *)(a1 + 272);
                  v42 = (v45 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                  v43 = *(_QWORD *)(v41 + 8LL * v42);
                  if ( v65 == v43 )
                  {
LABEL_42:
                    if ( v62 == ++v37 )
                      break;
                  }
                  else
                  {
                    while ( v43 != -8 )
                    {
                      v42 = v39 & (v40 + v42);
                      v43 = *(_QWORD *)(v41 + 8LL * v42);
                      if ( v65 == v43 )
                        goto LABEL_42;
                      ++v40;
                    }
LABEL_44:
                    for ( j = *(_QWORD *)(v65 + 8); j; j = *(_QWORD *)(j + 8) )
                    {
                      if ( (unsigned __int8)(*((_BYTE *)sub_1648700(j) + 16) - 25) <= 9u )
                        break;
                    }
                    if ( !sub_38520A0(j, 0, &v63, &v65) )
                      goto LABEL_42;
                    v49 = (unsigned int)v67;
                    if ( (unsigned int)v67 >= HIDWORD(v67) )
                    {
                      sub_16CD150((__int64)&v66, v68, 0, 8, v47, v48);
                      v49 = (unsigned int)v67;
                    }
                    ++v37;
                    v66[v49] = v44;
                    LODWORD(v67) = v67 + 1;
                    if ( v62 == v37 )
                      break;
                  }
                }
              }
            }
            v16 = v67;
            v15 = v66;
          }
          while ( (_DWORD)v67 );
LABEL_51:
          v5 = v58;
          if ( v15 == v68 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v15);
          v5 = v58 + 1;
          if ( v61 == v58 + 1 )
            return;
        }
      }
    }
  }
}
