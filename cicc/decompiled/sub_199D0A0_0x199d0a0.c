// Function: sub_199D0A0
// Address: 0x199d0a0
//
char __fastcall sub_199D0A0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v14; // rsi
  int v15; // eax
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rcx
  int v20; // r8d
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r14
  int v24; // eax
  __int64 v25; // rsi
  int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // r8
  int v29; // r9d
  unsigned __int64 v30; // rbx
  int v31; // eax
  int v32; // edx
  int v33; // ebx
  __int64 v34; // r12
  __int64 v35; // rbx
  unsigned __int64 v36; // rcx
  unsigned int v37; // edx
  unsigned int v38; // eax
  unsigned int v39; // eax
  unsigned int v40; // ecx
  unsigned int v41; // eax
  int v42; // edx
  int v43; // eax
  unsigned __int64 v45; // [rsp+10h] [rbp-70h]
  unsigned int v46; // [rsp+1Ch] [rbp-64h]
  int v47; // [rsp+20h] [rbp-60h]
  int v48; // [rsp+24h] [rbp-5Ch]
  __int64 *v50; // [rsp+30h] [rbp-50h]
  __int64 v52; // [rsp+40h] [rbp-40h] BYREF
  int v53; // [rsp+48h] [rbp-38h]

  v14 = *(_QWORD *)(a3 + 80);
  v48 = *(_DWORD *)(a1 + 8);
  v46 = *(_DWORD *)(a1 + 4);
  v47 = *(_DWORD *)(a1 + 16);
  if ( !v14 )
    goto LABEL_9;
  v15 = *(_DWORD *)(a6 + 24);
  if ( v15 )
  {
    v16 = v15 - 1;
    v17 = *(_QWORD *)(a6 + 8);
    LODWORD(v18) = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v19 = *(_QWORD *)(v17 + 8LL * (unsigned int)v18);
    if ( v14 == v19 )
    {
LABEL_12:
      *(_QWORD *)a1 = -1;
      *(_QWORD *)(a1 + 8) = -1;
      *(_QWORD *)(a1 + 16) = -1;
      *(_QWORD *)(a1 + 24) = -1;
      return v18;
    }
    v20 = 1;
    while ( v19 != -8 )
    {
      LODWORD(v18) = v16 & (v20 + v18);
      v19 = *(_QWORD *)(v17 + 8LL * (unsigned int)v18);
      if ( v14 == v19 )
        goto LABEL_12;
      ++v20;
    }
  }
  LOBYTE(v18) = (unsigned __int8)sub_199CF40(a1, v14, a4, a5, a7, a8, a10, (__int64)a2);
  if ( *(_DWORD *)(a1 + 4) != -1 )
  {
LABEL_9:
    v21 = *(__int64 **)(a3 + 32);
    v22 = *(unsigned int *)(a3 + 40);
    v23 = v21;
    v50 = &v21[v22];
    if ( v21 != v50 )
    {
      do
      {
        v24 = *(_DWORD *)(a6 + 24);
        v25 = *v23;
        if ( v24 )
        {
          v26 = v24 - 1;
          v27 = *(_QWORD *)(a6 + 8);
          LODWORD(v18) = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v28 = *(_QWORD *)(v27 + 8LL * (unsigned int)v18);
          if ( v25 == v28 )
            goto LABEL_12;
          v29 = 1;
          while ( v28 != -8 )
          {
            LODWORD(v18) = v26 & (v29 + v18);
            v28 = *(_QWORD *)(v27 + 8LL * (unsigned int)v18);
            if ( v25 == v28 )
              goto LABEL_12;
            ++v29;
          }
        }
        LOBYTE(v18) = (unsigned __int8)sub_199CF40(a1, v25, a4, a5, a7, a8, a10, (__int64)a2);
        if ( *(_DWORD *)(a1 + 4) == -1 )
          return v18;
        ++v23;
      }
      while ( v50 != v23 );
      v22 = *(unsigned int *)(a3 + 40);
    }
    v30 = v22 - ((*(_QWORD *)(a3 + 80) == 0) - 1LL);
    if ( v30 <= 1 )
    {
      v32 = *(_DWORD *)(a1 + 16);
    }
    else
    {
      v31 = 1;
      if ( *(_QWORD *)(a3 + 24) )
        v31 = (unsigned __int8)sub_19937D0(a2, a9, a3) + 1;
      v32 = *(_DWORD *)(a1 + 16) + v30 - v31;
      *(_DWORD *)(a1 + 16) = v32;
    }
    v33 = 0;
    *(_DWORD *)(a1 + 16) = v32 - ((*(_QWORD *)(a3 + 88) == 0) - 1);
    if ( *(_QWORD *)(a3 + 24) )
    {
      if ( sub_1993620(
             a2,
             *(_QWORD *)(a9 + 712),
             *(_QWORD *)(a9 + 720),
             *(_DWORD *)(a9 + 32),
             *(_QWORD *)(a9 + 40),
             *(_DWORD *)(a9 + 48),
             *(_QWORD *)a3,
             *(_QWORD *)(a3 + 8),
             *(_BYTE *)(a3 + 16),
             *(_QWORD *)(a3 + 24)) )
      {
        if ( *(_DWORD *)(a9 + 32) == 2 )
        {
          v33 = sub_14A2C80(
                  a2,
                  *(_QWORD *)(a9 + 40),
                  *(_QWORD *)a3,
                  *(_QWORD *)(a3 + 8) + *(_QWORD *)(a9 + 712),
                  *(_BYTE *)(a3 + 16),
                  *(_QWORD *)(a3 + 24));
          v43 = sub_14A2C80(
                  a2,
                  *(_QWORD *)(a9 + 40),
                  *(_QWORD *)a3,
                  *(_QWORD *)(a3 + 8) + *(_QWORD *)(a9 + 720),
                  *(_BYTE *)(a3 + 16),
                  *(_QWORD *)(a3 + 24));
          if ( v33 < v43 )
            v33 = v43;
        }
      }
      else
      {
        v33 = *(_QWORD *)(a3 + 24) != 1;
      }
    }
    *(_DWORD *)(a1 + 28) += v33;
    v34 = *(_QWORD *)(a9 + 56);
    v18 = *(unsigned int *)(a9 + 64);
    if ( v34 != v34 + 80 * v18 )
    {
      v35 = v34 + 80 * v18;
      while ( 1 )
      {
        v36 = *(_QWORD *)(a3 + 8) + *(_QWORD *)(v34 + 72);
        if ( *(_QWORD *)a3 )
        {
          *(_DWORD *)(a1 + 20) += 64;
          if ( *(_DWORD *)(a9 + 32) != 2 || !v36 )
            goto LABEL_30;
        }
        else
        {
          if ( !v36 )
            goto LABEL_30;
          v52 = *(_QWORD *)(a3 + 8) + *(_QWORD *)(v34 + 72);
          v45 = v36;
          v53 = 64;
          LODWORD(v18) = sub_1997E70((__int64)&v52);
          *(_DWORD *)(a1 + 20) += v18;
          v36 = v45;
          if ( *(_DWORD *)(a9 + 32) != 2 )
            goto LABEL_30;
        }
        LOBYTE(v18) = sub_14A2A90(
                        a2,
                        *(_QWORD *)(a9 + 40),
                        *(_QWORD *)a3,
                        v36,
                        *(_BYTE *)(a3 + 16),
                        *(_QWORD *)(a3 + 24));
        if ( (_BYTE)v18 )
        {
LABEL_30:
          v34 += 80;
          if ( v35 == v34 )
            break;
        }
        else
        {
          v34 += 80;
          ++*(_DWORD *)(a1 + 16);
          if ( v35 == v34 )
            break;
        }
      }
    }
    if ( byte_4FB1DC0 )
    {
      v37 = sub_14A3140(a2, 0) - 1;
      v38 = *(_DWORD *)(a1 + 4);
      if ( v37 < v38 )
      {
        v39 = *(_DWORD *)a1 + v38;
        v40 = v39 - v37;
        v41 = v39 - v46;
        if ( v46 <= v37 )
          v41 = v40;
        *(_DWORD *)a1 = v41;
      }
      if ( *(_DWORD *)(a9 + 32) == 3
        && (*(_QWORD *)(a3 + 88) || *(_QWORD *)(a3 + 8) || *(_DWORD *)(a3 + 40) != 1 || *(_QWORD *)(a3 + 80))
        && !(unsigned __int8)sub_14A2AD0((__int64)a2) )
      {
        v42 = *(_DWORD *)a1 + 1;
      }
      else
      {
        v42 = *(_DWORD *)a1;
      }
      LODWORD(v18) = v42 + *(_DWORD *)(a1 + 8) - v48;
      *(_DWORD *)a1 = v18;
      if ( *(_DWORD *)(a9 + 32) != 3 )
      {
        LODWORD(v18) = *(_DWORD *)(a1 + 16) - v47 + v18;
        *(_DWORD *)a1 = v18;
      }
    }
  }
  return v18;
}
