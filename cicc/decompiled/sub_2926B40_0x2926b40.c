// Function: sub_2926B40
// Address: 0x2926b40
//
void __fastcall sub_2926B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // r14d
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  unsigned int v14; // r14d
  unsigned __int64 v15; // rcx
  __int64 v16; // r14
  unsigned int v17; // r14d
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // ecx
  char v21; // r11
  int v22; // r11d
  __int64 v23; // r10
  unsigned int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rdi
  _QWORD *v27; // rbx
  __int64 v28; // rax
  int v29; // eax
  _BOOL8 v30; // r8
  unsigned __int64 v31; // rcx
  __int64 *v32; // rdx
  __int64 v33; // rsi
  unsigned int v34; // eax
  char v35; // di
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned int v38; // ecx
  __int64 v39; // rax
  __int64 v40; // rcx
  unsigned int v41; // esi
  char v42; // al
  int v43; // eax
  int v44; // r10d
  unsigned int v45; // eax
  unsigned int v46; // eax
  int v47; // ecx
  unsigned int v48; // r8d
  __int64 *v49; // rax
  __int64 v50; // rax
  int v51; // eax
  int v52; // r10d
  unsigned int v53; // [rsp+8h] [rbp-68h]
  unsigned __int64 v54; // [rsp+8h] [rbp-68h]
  unsigned __int64 v55; // [rsp+8h] [rbp-68h]
  int v56; // [rsp+8h] [rbp-68h]
  unsigned __int64 v57; // [rsp+10h] [rbp-60h]
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+10h] [rbp-60h]
  __int64 v61; // [rsp+10h] [rbp-60h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  int v63; // [rsp+18h] [rbp-58h]
  __int64 *v64; // [rsp+18h] [rbp-58h]
  __int64 v65; // [rsp+18h] [rbp-58h]
  __int64 *v66; // [rsp+28h] [rbp-48h] BYREF
  __int64 v67; // [rsp+30h] [rbp-40h] BYREF
  int v68; // [rsp+38h] [rbp-38h]

  v8 = a2;
  v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v10 = *(_QWORD *)(a2 + 32 * (2 - v9));
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 > 0x40 )
    {
      if ( v11 - (unsigned int)sub_C444A0(v10 + 24) <= 0x40 && !**(_QWORD **)(v10 + 24) )
        goto LABEL_12;
    }
    else if ( !*(_QWORD *)(v10 + 24) )
    {
      goto LABEL_12;
    }
  }
  else
  {
    v10 = 0;
  }
  if ( *(_BYTE *)(a1 + 572) )
  {
    v12 = *(_QWORD **)(a1 + 552);
    v13 = &v12[*(unsigned int *)(a1 + 564)];
    if ( v12 != v13 )
    {
      while ( a2 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_14;
      }
      return;
    }
  }
  else if ( sub_C8CA60(a1 + 544, a2) )
  {
    return;
  }
LABEL_14:
  if ( *(_BYTE *)(a1 + 344) )
  {
    v9 = *(_QWORD *)(a1 + 376);
    if ( !*(_BYTE *)v9 )
    {
      v14 = *(_DWORD *)(a1 + 360);
      v15 = *(_QWORD *)(a1 + 368);
      a6 = a1 + 352;
      if ( v14 > 0x40 )
      {
        v54 = *(_QWORD *)(a1 + 368);
        v58 = *(_QWORD *)(a1 + 376);
        v34 = sub_C444A0(a1 + 352);
        a6 = a1 + 352;
        v9 = v58;
        a5 = v34;
        v15 = v54;
        if ( v14 - v34 <= 0x40 )
        {
          a5 = **(_QWORD **)(a1 + 352);
          if ( v54 > a5 )
          {
LABEL_18:
            v16 = v15 - a5;
            if ( v10 )
            {
              v17 = *(_DWORD *)(v10 + 32);
              if ( v17 > 0x40 )
              {
                v55 = a5;
                v59 = a6;
                v65 = v9;
                v43 = sub_C444A0(v10 + 24);
                v9 = v65;
                a6 = v59;
                v44 = v43;
                v45 = v17;
                a5 = v55;
                v16 = -1;
                if ( v45 - v44 <= 0x40 )
                  v16 = **(_QWORD **)(v10 + 24);
              }
              else
              {
                v16 = *(_QWORD *)(v10 + 24);
              }
            }
            v18 = **(_QWORD **)(a1 + 336);
            v19 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
            if ( *(_QWORD *)(v8 - 32 * v19) == v18 && v18 == *(_QWORD *)(v8 + 32 * (1 - v19)) )
            {
              v64 = (__int64 *)a6;
              v42 = sub_2919180(v8);
              a6 = (__int64)v64;
              if ( v42 )
              {
                v30 = 0;
                v31 = v16;
                v32 = v64;
                v33 = v8;
                goto LABEL_28;
              }
LABEL_12:
              sub_2916B30(a1, v8, (__int64 *)v9, a4, a5, a6);
              return;
            }
            v20 = *(_DWORD *)(v9 + 32);
            v21 = *(_BYTE *)(a1 + 392);
            v67 = v8;
            v68 = v20;
            v22 = v21 & 1;
            if ( v22 )
            {
              v23 = a1 + 400;
              v24 = ((unsigned __int8)((unsigned int)v8 >> 4) ^ (unsigned __int8)((unsigned int)v8 >> 9)) & 3;
              v25 = a1
                  + 400
                  + 16LL * (((unsigned __int8)((unsigned int)v8 >> 4) ^ (unsigned __int8)((unsigned int)v8 >> 9)) & 3);
              v26 = *(_QWORD *)v25;
              if ( *(_QWORD *)v25 == v8 )
                goto LABEL_24;
              v63 = 3;
            }
            else
            {
              v41 = *(_DWORD *)(a1 + 408);
              if ( !v41 )
              {
                v46 = *(_DWORD *)(a1 + 392);
                ++*(_QWORD *)(a1 + 384);
                v66 = 0;
                v47 = (v46 >> 1) + 1;
                goto LABEL_55;
              }
              v23 = *(_QWORD *)(a1 + 400);
              v63 = v41 - 1;
              v24 = (v41 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
              v25 = v23 + 16LL * v24;
              v26 = *(_QWORD *)v25;
              if ( v8 == *(_QWORD *)v25 )
              {
LABEL_24:
                a4 = 3LL * *(unsigned int *)(v25 + 8);
                v27 = (_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * a4);
                v28 = *(_QWORD *)(v8 + 32 * (3 - v19));
                v9 = *(unsigned int *)(v28 + 32);
                if ( (unsigned int)v9 <= 0x40 )
                {
                  if ( *(_QWORD *)(v28 + 24) )
                    goto LABEL_26;
                }
                else
                {
                  v53 = *(_DWORD *)(v28 + 32);
                  v57 = a5;
                  v62 = a6;
                  v29 = sub_C444A0(v28 + 24);
                  v9 = v53;
                  a6 = v62;
                  a5 = v57;
                  if ( v53 != v29 )
                  {
LABEL_26:
                    v27[2] &= ~4uLL;
                    v30 = 0;
LABEL_27:
                    v31 = v16;
                    v32 = (__int64 *)a6;
                    v33 = a2;
LABEL_28:
                    sub_2916EE0(a1, v33, v32, v31, v30);
                    return;
                  }
                }
                if ( *v27 == a5 )
                {
                  v27[2] &= 7uLL;
                  goto LABEL_12;
                }
                goto LABEL_26;
              }
            }
            v56 = 1;
            v60 = 0;
            while ( v26 != -4096 )
            {
              if ( !v60 )
              {
                if ( v26 != -8192 )
                  v25 = 0;
                v60 = v25;
              }
              v24 = v63 & (v56 + v24);
              v25 = v23 + 16LL * v24;
              v26 = *(_QWORD *)v25;
              if ( v8 == *(_QWORD *)v25 )
                goto LABEL_24;
              ++v56;
            }
            if ( v60 )
              v25 = v60;
            v46 = *(_DWORD *)(a1 + 392);
            ++*(_QWORD *)(a1 + 384);
            v66 = (__int64 *)v25;
            v47 = (v46 >> 1) + 1;
            if ( (_BYTE)v22 )
            {
              v48 = 12;
              v41 = 4;
LABEL_56:
              if ( 4 * v47 >= v48 )
              {
                v61 = a6;
                v41 *= 2;
              }
              else
              {
                if ( v41 - *(_DWORD *)(a1 + 396) - v47 > v41 >> 3 )
                {
LABEL_58:
                  *(_DWORD *)(a1 + 392) = (2 * (v46 >> 1) + 2) | v46 & 1;
                  v49 = v66;
                  if ( *v66 != -4096 )
                    --*(_DWORD *)(a1 + 396);
                  *v49 = v8;
                  v30 = v10 != 0;
                  *((_DWORD *)v49 + 2) = v68;
                  goto LABEL_27;
                }
                v61 = a6;
              }
              sub_FA7CC0(a1 + 384, v41);
              sub_2921A30(a1 + 384, &v67, &v66);
              v8 = v67;
              v46 = *(_DWORD *)(a1 + 392);
              a6 = v61;
              goto LABEL_58;
            }
            v41 = *(_DWORD *)(a1 + 408);
LABEL_55:
            v48 = 3 * v41;
            goto LABEL_56;
          }
        }
      }
      else
      {
        a5 = *(_QWORD *)(a1 + 352);
        if ( v15 > a5 )
          goto LABEL_18;
      }
      v35 = *(_BYTE *)(a1 + 392) & 1;
      if ( (*(_BYTE *)(a1 + 392) & 1) != 0 )
      {
        v37 = a1 + 400;
        a5 = 3;
      }
      else
      {
        v36 = *(unsigned int *)(a1 + 408);
        v37 = *(_QWORD *)(a1 + 400);
        if ( !(_DWORD)v36 )
          goto LABEL_69;
        a5 = (unsigned int)(v36 - 1);
      }
      v38 = a5 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v39 = v37 + 16LL * v38;
      a6 = *(_QWORD *)v39;
      if ( v8 == *(_QWORD *)v39 )
      {
LABEL_35:
        v40 = 64;
        if ( !v35 )
          v40 = 16LL * *(unsigned int *)(a1 + 408);
        a4 = v37 + v40;
        if ( v39 != a4 )
        {
          a4 = 3LL * *(unsigned int *)(v39 + 8);
          *(_QWORD *)(*(_QWORD *)(v9 + 24) + 24LL * *(unsigned int *)(v39 + 8) + 16) &= 7uLL;
        }
        goto LABEL_12;
      }
      v51 = 1;
      while ( a6 != -4096 )
      {
        v52 = v51 + 1;
        v38 = a5 & (v51 + v38);
        v39 = v37 + 16LL * v38;
        a6 = *(_QWORD *)v39;
        if ( v8 == *(_QWORD *)v39 )
          goto LABEL_35;
        v51 = v52;
      }
      if ( v35 )
      {
        v50 = 64;
        goto LABEL_70;
      }
      v36 = *(unsigned int *)(a1 + 408);
LABEL_69:
      v50 = 16 * v36;
LABEL_70:
      v39 = v37 + v50;
      goto LABEL_35;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a2;
  }
}
