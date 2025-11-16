// Function: sub_1F06050
// Address: 0x1f06050
//
__int64 __fastcall sub_1F06050(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  int v5; // eax
  int v6; // r9d
  int v7; // r14d
  unsigned int v8; // esi
  unsigned int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // r12
  _DWORD *v12; // rax
  __int64 v13; // rax
  int v14; // r14d
  __int64 v15; // r12
  __int64 v16; // rax
  int v17; // r15d
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 result; // rax
  unsigned int v21; // r10d
  unsigned int v22; // ecx
  __int64 v23; // rsi
  _DWORD *v24; // rdi
  __int64 v25; // rdx
  int v26; // r12d
  __int64 v27; // r14
  __int64 v28; // r15
  __int64 v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rcx
  int v32; // r8d
  unsigned __int64 v33; // r9
  int v34; // edx
  int v35; // ecx
  __int64 v36; // r15
  __int64 v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // r10
  __int64 v41; // rcx
  int v42; // eax
  __int64 v43; // rcx
  unsigned __int64 v44; // r9
  __int64 v45; // r10
  void (*v46)(); // rax
  unsigned int v47; // r11d
  unsigned int v48; // esi
  _DWORD *v49; // rcx
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // [rsp+10h] [rbp-80h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  int v55; // [rsp+20h] [rbp-70h]
  __int64 v56; // [rsp+28h] [rbp-68h]
  int v58; // [rsp+34h] [rbp-5Ch]
  unsigned __int64 v60; // [rsp+40h] [rbp-50h] BYREF
  int v61; // [rsp+48h] [rbp-48h]
  int v62; // [rsp+4Ch] [rbp-44h]
  unsigned __int64 v63; // [rsp+50h] [rbp-40h] BYREF
  __int64 v64; // [rsp+58h] [rbp-38h]

  v56 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(v56 + 32) + 40LL * a3;
  v58 = *(_DWORD *)(v4 + 8);
  if ( *(_BYTE *)(a1 + 914) )
  {
    if ( (*(_DWORD *)v4 & 0xFFF00) == 0 || (*(_BYTE *)(v4 + 4) & 1) != 0 )
    {
      v5 = sub_1F04440(a1, (_DWORD *)v4);
      v6 = -1;
      v7 = v5;
    }
    else
    {
      v6 = sub_1F04440(a1, (_DWORD *)v4);
      v7 = v6;
    }
    *(_BYTE *)(v4 + 4) &= ~1u;
  }
  else
  {
    v6 = -1;
    v7 = -1;
  }
  if ( (((*(_BYTE *)(v4 + 3) & 0x40) != 0) & (*(_BYTE *)(v4 + 3) >> 4)) == 0 )
  {
    v8 = *(_DWORD *)(a1 + 1688);
    v52 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
    v9 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1952) + (v58 & 0x7FFFFFFF));
    if ( v9 < v8 )
    {
      v10 = *(_QWORD *)(a1 + 1680);
      while ( 1 )
      {
        v11 = v9;
        v12 = (_DWORD *)(v10 + 32LL * v9);
        if ( (v58 & 0x7FFFFFFF) == (*v12 & 0x7FFFFFFF) )
        {
          v13 = (unsigned int)v12[6];
          if ( (_DWORD)v13 != -1 && *(_DWORD *)(v10 + 32 * v13 + 28) == -1 )
            break;
        }
        v9 += 256;
        if ( v8 <= v9 )
          goto LABEL_23;
      }
      if ( v9 != -1 )
      {
        v55 = v7;
        v14 = v6;
        while ( 1 )
        {
          v15 = 32 * v11;
          v16 = v10 + v15;
          v17 = *(_DWORD *)(v10 + v15 + 4);
          if ( (v17 & v14) != 0 )
          {
            if ( (v17 & v55) != 0 )
            {
              v40 = *(_QWORD *)(v16 + 8);
              v41 = *(_QWORD *)(v40 + 8);
              v63 = a2 & 0xFFFFFFFFFFFFFFF9LL;
              v64 = (unsigned int)v58 | 0x100000000LL;
              v54 = v40;
              v42 = sub_1F4BB70(a1 + 632, v56, a3, v41, *(unsigned int *)(v16 + 16));
              v45 = v54;
              HIDWORD(v64) = v42;
              v46 = *(void (**)())(*(_QWORD *)v52 + 208LL);
              if ( v46 != nullsub_681 )
              {
                ((void (__fastcall *)(__int64, __int64, __int64, unsigned __int64 *))v46)(v52, a2, v54, &v63);
                v45 = v54;
              }
              sub_1F01A00(v45, (__int64)&v63, 1, v43, (int)&v63, v44);
              v10 = *(_QWORD *)(a1 + 1680);
              v16 = v10 + v15;
            }
            if ( (v17 & ~v14) != 0 )
            {
              *(_DWORD *)(v16 + 4) = v17 & ~v14;
              v9 = *(_DWORD *)(*(_QWORD *)(a1 + 1680) + v15 + 28);
            }
            else
            {
              v35 = -1;
              v36 = *(unsigned int *)(v16 + 24);
              v37 = 32 * v36;
              v38 = v10 + 32 * v36;
              if ( v38 != v16 )
              {
                v39 = *(unsigned int *)(v16 + 28);
                if ( *(_DWORD *)(v38 + 28) == -1 )
                {
                  *(_BYTE *)(*(_QWORD *)(a1 + 1952) + (*(_DWORD *)v16 & 0x7FFFFFFF)) = v39;
                  *(_DWORD *)(*(_QWORD *)(a1 + 1680) + 32LL * *(unsigned int *)(v16 + 28) + 24) = *(_DWORD *)(v16 + 24);
                  v35 = *(_DWORD *)(v16 + 28);
                  v16 = v15 + *(_QWORD *)(a1 + 1680);
                }
                else if ( (_DWORD)v39 == -1 )
                {
                  v47 = *(_DWORD *)(a1 + 1688);
                  v48 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1952) + (*(_DWORD *)v16 & 0x7FFFFFFF));
                  if ( v48 < v47 )
                  {
                    while ( 1 )
                    {
                      v49 = (_DWORD *)(v10 + 32LL * v48);
                      if ( (*(_DWORD *)v16 & 0x7FFFFFFF) == (*v49 & 0x7FFFFFFF) )
                      {
                        v50 = (unsigned int)v49[6];
                        if ( (_DWORD)v50 != -1 && *(_DWORD *)(v10 + 32 * v50 + 28) == -1 )
                          break;
                      }
                      v48 += 256;
                      if ( v47 <= v48 )
                        goto LABEL_69;
                    }
                  }
                  else
                  {
LABEL_69:
                    v49 = (_DWORD *)(v10 + 0x1FFFFFFFE0LL);
                  }
                  v49[6] = v36;
                  *(_DWORD *)(*(_QWORD *)(a1 + 1680) + v37 + 28) = *(_DWORD *)(v16 + 28);
                  v51 = *(_QWORD *)(a1 + 1680);
                  v35 = *(_DWORD *)(v51 + 32LL * *(unsigned int *)(v16 + 24) + 28);
                  v16 = v51 + v15;
                }
                else
                {
                  *(_DWORD *)(v10 + 32 * v39 + 24) = v36;
                  v35 = *(_DWORD *)(v16 + 28);
                  *(_DWORD *)(*(_QWORD *)(a1 + 1680) + v37 + 28) = v35;
                  v16 = v15 + *(_QWORD *)(a1 + 1680);
                }
              }
              *(_DWORD *)(v16 + 24) = -1;
              *(_DWORD *)(*(_QWORD *)(a1 + 1680) + v15 + 28) = *(_DWORD *)(a1 + 1968);
              *(_DWORD *)(a1 + 1968) = v9;
              v9 = v35;
              ++*(_DWORD *)(a1 + 1972);
            }
            if ( v9 == -1 )
            {
LABEL_22:
              v7 = v55;
              break;
            }
          }
          else
          {
            v9 = *(_DWORD *)(v16 + 28);
            if ( v9 == -1 )
              goto LABEL_22;
          }
          v10 = *(_QWORD *)(a1 + 1680);
          v11 = v9;
        }
      }
    }
  }
LABEL_23:
  v18 = *(_QWORD *)(a1 + 40);
  if ( v58 < 0 )
    v19 = *(_QWORD *)(*(_QWORD *)(v18 + 24) + 16LL * (v58 & 0x7FFFFFFF) + 8);
  else
    v19 = *(_QWORD *)(*(_QWORD *)(v18 + 272) + 8LL * (unsigned int)v58);
  if ( !v19
    || (*(_BYTE *)(v19 + 3) & 0x10) == 0 && ((v19 = *(_QWORD *)(v19 + 32)) == 0 || (*(_BYTE *)(v19 + 3) & 0x10) == 0)
    || (result = *(_QWORD *)(v19 + 32)) != 0 && (*(_BYTE *)(result + 3) & 0x10) != 0 )
  {
    v53 = a1 + 1448;
    v21 = *(_DWORD *)(a1 + 1456);
    result = v58 & 0x7FFFFFFF;
    v22 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1656) + result);
    if ( v22 < v21 )
    {
      v23 = *(_QWORD *)(a1 + 1448);
      while ( 1 )
      {
        result = v22;
        v24 = (_DWORD *)(v23 + 24LL * v22);
        if ( (v58 & 0x7FFFFFFF) == (*v24 & 0x7FFFFFFF) )
        {
          v25 = (unsigned int)v24[4];
          if ( (_DWORD)v25 != -1 && *(_DWORD *)(v23 + 24 * v25 + 20) == -1 )
            break;
        }
        v22 += 256;
        if ( v21 <= v22 )
          goto LABEL_44;
      }
      if ( v22 != -1 )
      {
        v26 = v7;
        do
        {
          v28 = 24 * result;
          v27 = v23 + 24 * result;
          if ( (*(_DWORD *)(v27 + 4) & v26) != 0 )
          {
            v29 = *(_QWORD *)(v27 + 8);
            if ( a2 != v29 )
            {
              v62 = 0;
              v61 = v58;
              v30 = *(_QWORD *)(v29 + 8);
              v60 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
              v62 = sub_1F4BFE0(a1 + 632, v56, a3, v30);
              sub_1F01A00(v29, (__int64)&v60, 1, v31, v32, v33);
              v34 = *(_DWORD *)(v27 + 4);
              *(_QWORD *)(v27 + 8) = a2;
              *(_DWORD *)(v27 + 4) = v34 & v26;
              if ( (v34 & ~v26) != 0 )
              {
                HIDWORD(v63) = v34 & ~v26;
                v64 = v29;
                LODWORD(v63) = v58;
                sub_1E74F70(v53, (__int64)&v63);
              }
              v23 = *(_QWORD *)(a1 + 1448);
              v27 = v23 + v28;
            }
          }
          result = *(unsigned int *)(v27 + 20);
        }
        while ( (_DWORD)result != -1 );
        v7 = v26;
      }
    }
LABEL_44:
    if ( v7 )
    {
      v63 = __PAIR64__(v7, v58);
      v64 = a2;
      return sub_1E74F70(v53, (__int64)&v63);
    }
  }
  return result;
}
