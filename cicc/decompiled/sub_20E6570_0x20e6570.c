// Function: sub_20E6570
// Address: 0x20e6570
//
__int64 __fastcall sub_20E6570(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 result; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  unsigned int v7; // ebx
  __int64 v8; // rsi
  __int64 v9; // r13
  __int64 *v10; // rax
  _QWORD *v11; // rdi
  _WORD *v12; // rdx
  unsigned __int16 *v13; // rax
  int v14; // r11d
  unsigned __int16 *v15; // r13
  unsigned __int16 *v16; // r8
  unsigned __int16 *v17; // rdx
  unsigned __int16 v18; // ax
  unsigned __int16 v19; // si
  __int16 *v20; // r9
  __int16 *v21; // rdx
  __int16 *v22; // rcx
  __int16 v23; // cx
  __int64 v24; // r9
  _QWORD *v25; // rdx
  __int16 v26; // dx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  unsigned __int64 v31; // rcx
  __int64 v32; // rdx
  _QWORD *v33; // rax
  unsigned __int16 *v34; // rdx
  __int64 v35; // rdx
  __int16 v36; // ax
  __int64 v37; // rdi
  __int64 (*v38)(); // rax
  int v39; // edx
  __int64 v40; // rax
  unsigned __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rdx
  _QWORD *v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned __int16 *v48; // rax
  int v49; // edx
  unsigned __int64 v50; // rcx
  __int64 v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // [rsp+8h] [rbp-88h]
  char v54; // [rsp+17h] [rbp-79h]
  __int64 v55; // [rsp+20h] [rbp-70h]
  __int64 v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+38h] [rbp-58h]
  unsigned int v59; // [rsp+50h] [rbp-40h] BYREF
  __int64 v60; // [rsp+58h] [rbp-38h]

  v3 = *(_WORD *)(a2 + 46);
  if ( (v3 & 4) == 0 && (v3 & 8) != 0 )
  {
    if ( sub_1E15D00(a2, 0x10u, 1) )
      goto LABEL_4;
  }
  else if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) & 0x10LL) != 0 )
  {
LABEL_4:
    v54 = 1;
    goto LABEL_5;
  }
  v36 = *(_WORD *)(a2 + 46);
  if ( (v36 & 4) != 0 || (v36 & 8) == 0 )
    v54 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) & 0x8000000LL) != 0;
  else
    v54 = sub_1E15D00(a2, 0x8000000u, 1);
  if ( v54 )
    goto LABEL_4;
  v37 = *(_QWORD *)(a1 + 24);
  v38 = *(__int64 (**)())(*(_QWORD *)v37 + 656LL);
  if ( v38 != sub_1D918C0 )
    v54 = ((__int64 (__fastcall *)(__int64, __int64))v38)(v37, a2);
LABEL_5:
  result = a2;
  v5 = 0;
  v57 = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)v57 )
  {
    while ( 1 )
    {
      result = 40 * v5;
      v6 = 40 * v5 + *(_QWORD *)(a2 + 32);
      if ( !*(_BYTE *)v6 )
      {
        v7 = *(_DWORD *)(v6 + 8);
        if ( v7 )
          break;
      }
LABEL_7:
      if ( v57 == ++v5 )
        return result;
    }
    v8 = *(_QWORD *)(a2 + 16);
    v9 = 8LL * v7;
    if ( (unsigned int)v5 >= *(unsigned __int16 *)(v8 + 2) )
    {
      v10 = (__int64 *)(v9 + *(_QWORD *)(a1 + 72));
LABEL_12:
      *v10 = -1;
      goto LABEL_13;
    }
    v35 = sub_1F3AD60(*(_QWORD *)(a1 + 24), v8, v5, *(_QWORD **)(a1 + 32), *(_QWORD *)(a1 + 8));
    v10 = (__int64 *)(v9 + *(_QWORD *)(a1 + 72));
    if ( *v10 )
    {
      if ( !v35 || *v10 != v35 )
        goto LABEL_12;
    }
    else
    {
      if ( !v35 )
        goto LABEL_12;
      *v10 = v35;
    }
LABEL_13:
    v11 = *(_QWORD **)(a1 + 32);
    if ( !v11 )
      BUG();
    v56 = v11[1];
    v53 = 24LL * v7;
    v55 = v11[7];
    v14 = v7 * (*(_DWORD *)(v56 + v53 + 16) & 0xF);
    v12 = (_WORD *)(v55 + 2LL * (*(_DWORD *)(v56 + v53 + 16) >> 4));
    v13 = v12 + 1;
    LOWORD(v14) = *v12 + v14;
LABEL_15:
    v15 = v13;
    while ( 1 )
    {
      v16 = v15;
      if ( !v15 )
        break;
      v17 = (unsigned __int16 *)(v11[6] + 4LL * (unsigned __int16)v14);
      v18 = *v17;
      v19 = v17[1];
      if ( *v17 )
      {
        while ( 1 )
        {
          v20 = (__int16 *)(v55 + 2LL * *(unsigned int *)(v56 + 24LL * v18 + 8));
          while ( 1 )
          {
            v21 = v20;
            v22 = v20;
            if ( !v20 )
              break;
            while ( 1 )
            {
              if ( v7 != v18 )
              {
                v24 = *(_QWORD *)(a1 + 72);
                while ( 1 )
                {
                  v25 = (_QWORD *)(v24 + 8LL * v18);
                  if ( *v25 )
                  {
                    *v25 = -1;
                    *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * v7) = -1;
                    v24 = *(_QWORD *)(a1 + 72);
                  }
                  do
                  {
                    v26 = *v22;
                    if ( *v22 )
                    {
                      ++v22;
                      v18 += v26;
                    }
                    else if ( v19 )
                    {
                      v22 = (__int16 *)(v11[7] + 2LL * *(unsigned int *)(v11[1] + 24LL * v19 + 8));
                      v18 = v19;
                      v19 = 0;
                    }
                    else
                    {
                      v27 = *v16;
                      if ( !(_WORD)v27 )
                        goto LABEL_31;
                      v14 += v27;
                      ++v16;
                      v34 = (unsigned __int16 *)(v11[6] + 4LL * (unsigned __int16)v14);
                      v18 = *v34;
                      v19 = v34[1];
                      v22 = (__int16 *)(v11[7] + 2LL * *(unsigned int *)(v11[1] + 24LL * *v34 + 8));
                    }
                  }
                  while ( v7 == v18 );
                }
              }
              v23 = *v21;
              v20 = 0;
              ++v21;
              if ( !v23 )
                break;
              v18 += v23;
              v22 = v21;
              if ( !v21 )
                goto LABEL_23;
            }
          }
LABEL_23:
          v18 = v19;
          if ( !v19 )
            break;
          v19 = 0;
        }
      }
      v39 = *v15;
      v13 = 0;
      ++v15;
      v14 += v39;
      if ( !(_WORD)v39 )
        goto LABEL_15;
    }
    v24 = *(_QWORD *)(a1 + 72);
LABEL_31:
    if ( *(_QWORD *)(v24 + 8LL * v7) != -1 )
    {
      v59 = v7;
      v60 = v6;
      sub_20E64D0(a1 + 96, &v59);
    }
    result = *(_QWORD *)(a2 + 32) + 40 * v5;
    if ( !*(_BYTE *)result && (*(_BYTE *)(result + 3) & 0x10) != 0 && (*(_WORD *)(result + 2) & 0xFF0) != 0 )
    {
      result = *(_QWORD *)(a1 + 72);
      if ( *(_QWORD *)(result + 8LL * v7) == -1 )
      {
        v40 = *(_QWORD *)(a1 + 32);
        if ( !v40 )
          BUG();
        v41 = v7;
        v42 = *(_QWORD *)(v40 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v40 + 8) + v53 + 4);
        while ( 1 )
        {
          v43 = v42;
          if ( !v42 )
            break;
          while ( 1 )
          {
            v43 += 2;
            v44 = (_QWORD *)(*(_QWORD *)(a1 + 192) + ((v41 >> 3) & 0x1FF8));
            *v44 |= 1LL << v41;
            v45 = *(unsigned __int16 *)(v43 - 2);
            v42 = 0;
            if ( !(_WORD)v45 )
              break;
            v41 = (unsigned int)(v45 + v41);
            if ( !v43 )
              goto LABEL_73;
          }
        }
LABEL_73:
        v46 = *(_QWORD *)(a1 + 32);
        if ( !v46 )
          BUG();
        v47 = 0;
        v48 = (unsigned __int16 *)(*(_QWORD *)(v46 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v46 + 8) + v53 + 8));
        v49 = *v48;
        result = (__int64)(v48 + 1);
        v50 = v49 + v7;
        if ( (_WORD)v49 )
          v47 = result;
        while ( 1 )
        {
          v51 = v47;
          if ( !v47 )
            break;
          while ( 1 )
          {
            v51 += 2;
            v52 = (_QWORD *)(*(_QWORD *)(a1 + 192) + ((v50 >> 3) & 0x1FF8));
            *v52 |= 1LL << v50;
            result = *(unsigned __int16 *)(v51 - 2);
            v47 = 0;
            if ( !(_WORD)result )
              break;
            v50 = (unsigned int)(result + v50);
            if ( !v51 )
              goto LABEL_37;
          }
        }
      }
    }
LABEL_37:
    if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
    {
      if ( v54 )
      {
        result = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * (v7 >> 6)) & (1LL << v7);
        if ( !result )
        {
          v28 = *(_QWORD *)(a1 + 32);
          if ( !v28 )
            BUG();
          v29 = *(_QWORD *)(v28 + 8);
          result = *(_QWORD *)(v28 + 56);
          v30 = result + 2LL * *(unsigned int *)(v29 + v53 + 4);
          v31 = (unsigned __int16)v7;
          while ( 1 )
          {
            v32 = v30;
            if ( !v30 )
              break;
            while ( 1 )
            {
              v32 += 2;
              v33 = (_QWORD *)(*(_QWORD *)(a1 + 192) + ((v31 >> 3) & 0x1FF8));
              *v33 |= 1LL << v31;
              result = *(unsigned __int16 *)(v32 - 2);
              v30 = 0;
              if ( !(_WORD)result )
                break;
              v31 = (unsigned int)(result + v31);
              if ( !v32 )
                goto LABEL_7;
            }
          }
        }
      }
    }
    goto LABEL_7;
  }
  return result;
}
