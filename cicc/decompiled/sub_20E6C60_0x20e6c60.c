// Function: sub_20E6C60
// Address: 0x20e6c60
//
__int64 __fastcall sub_20E6C60(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r15
  _QWORD *v4; // rbx
  __int64 (*v5)(void); // rax
  __int64 result; // rax
  __int64 v7; // r13
  _QWORD *v8; // r15
  unsigned __int8 *v9; // rbx
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned int v13; // r14d
  __int64 v14; // rsi
  __int64 v15; // r11
  __int64 *v16; // r10
  _QWORD *v17; // rdx
  __int64 v18; // rsi
  unsigned __int16 v19; // r10
  __int64 v20; // rcx
  unsigned int v21; // edi
  __int16 v22; // r14
  _WORD *v23; // rdi
  unsigned __int16 *v24; // r8
  unsigned __int16 v25; // r14
  unsigned __int16 *v26; // rdi
  unsigned __int16 *v27; // r8
  unsigned __int16 *v28; // rax
  unsigned __int16 v29; // r11
  __int16 *v30; // r10
  _DWORD *v31; // rsi
  __int16 v32; // cx
  __int64 v33; // rax
  int v34; // r12d
  __int64 v35; // rax
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rcx
  _QWORD *v40; // rax
  unsigned __int16 v41; // r15
  _QWORD *v42; // r12
  __int16 v43; // ax
  __int16 *v44; // r14
  __int64 v45; // rdx
  unsigned __int16 *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned __int16 *v52; // rax
  int v53; // ecx
  unsigned __int16 *v54; // rax
  int v55; // r12d
  int v56; // edx
  unsigned __int16 v57; // r11
  __int64 v58; // [rsp+8h] [rbp-68h]
  unsigned int v59; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+10h] [rbp-60h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+20h] [rbp-50h]
  unsigned int v66; // [rsp+30h] [rbp-40h] BYREF
  __int64 v67; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v4 = (_QWORD *)a1;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 24) + 656LL);
  if ( v5 != sub_1D918C0 )
  {
    result = v5();
    if ( (_BYTE)result )
      goto LABEL_13;
  }
  result = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)result )
  {
    v7 = 0;
    v8 = (_QWORD *)a1;
    v61 = 40 * result;
    do
    {
      while ( 1 )
      {
        v9 = (unsigned __int8 *)(v7 + *(_QWORD *)(a2 + 32));
        result = *v9;
        if ( (_BYTE)result == 12 )
        {
          v33 = v8[4];
          v66 = 0;
          v34 = *(_DWORD *)(v33 + 16);
          v35 = 0;
          if ( v34 )
          {
            do
            {
              while ( 1 )
              {
                v36 = *(_DWORD *)(*((_QWORD *)v9 + 3) + 4LL * ((unsigned int)v35 >> 5));
                if ( !_bittest(&v36, v35) )
                  break;
                v35 = v66 + 1;
                v66 = v35;
                if ( (_DWORD)v35 == v34 )
                  goto LABEL_39;
              }
              *(_DWORD *)(v8[21] + 4 * v35) = a3;
              *(_DWORD *)(v8[18] + 4LL * v66) = -1;
              *(_QWORD *)(v8[24] + 8LL * (v66 >> 6)) &= ~(1LL << v66);
              *(_QWORD *)(v8[9] + 8LL * v66) = 0;
              sub_20E6B90(v8 + 12, &v66);
              v35 = v66 + 1;
              v66 = v35;
            }
            while ( (_DWORD)v35 != v34 );
          }
LABEL_39:
          result = *v9;
        }
        if ( !(_BYTE)result )
        {
          result = *((unsigned int *)v9 + 2);
          v59 = result;
          if ( (_DWORD)result )
          {
            if ( (v9[3] & 0x10) != 0 )
            {
              result = v7 + *(_QWORD *)(a2 + 32);
              if ( *(_BYTE *)result || (*(_BYTE *)(result + 3) & 0x10) == 0 || (*(_WORD *)(result + 2) & 0xFF0) == 0 )
                break;
            }
          }
        }
        v7 += 40;
        if ( v61 == v7 )
          goto LABEL_12;
      }
      v37 = v8[4];
      if ( !v37 )
        BUG();
      v58 = v7;
      v38 = (1LL << v59) & *(_QWORD *)(v8[24] + 8LL * (v59 >> 6));
      v39 = *(_QWORD *)(v37 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v37 + 8) + 24LL * v59 + 4);
      v40 = v8;
      v41 = v59;
      v42 = v40;
LABEL_43:
      v44 = (__int16 *)v39;
      while ( v44 )
      {
        v45 = v42[21];
        v66 = v41;
        *(_DWORD *)(v45 + 4LL * v41) = a3;
        *(_DWORD *)(v42[18] + 4LL * v66) = -1;
        *(_QWORD *)(v42[9] + 8LL * v66) = 0;
        sub_20E6B90(v42 + 12, &v66);
        if ( !v38 )
          *(_QWORD *)(v42[24] + 8LL * (v66 >> 6)) &= ~(1LL << v66);
        v43 = *v44;
        v39 = 0;
        ++v44;
        v41 += v43;
        if ( !v43 )
          goto LABEL_43;
      }
      v50 = v42[4];
      v8 = v42;
      if ( !v50 )
        BUG();
      v51 = 0;
      v52 = (unsigned __int16 *)(*(_QWORD *)(v50 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v50 + 8) + 24LL * v59 + 8));
      v53 = *v52;
      v54 = v52 + 1;
      v55 = v53 + (unsigned __int16)v59;
      if ( (_WORD)v53 )
        v51 = (__int64)v54;
LABEL_61:
      result = v51;
      while ( result )
      {
        result += 2;
        v51 = 0;
        *(_QWORD *)(v8[9] + 8LL * (unsigned __int16)v55) = -1;
        v56 = *(unsigned __int16 *)(result - 2);
        v55 += v56;
        if ( !(_WORD)v56 )
          goto LABEL_61;
      }
      v7 += 40;
    }
    while ( v61 != v58 + 40 );
LABEL_12:
    v4 = v8;
    v3 = a2;
LABEL_13:
    v10 = *(unsigned int *)(v3 + 40);
    v11 = 0;
    if ( !(_DWORD)v10 )
      return result;
    while ( 1 )
    {
      result = 5 * v11;
      v12 = *(_QWORD *)(v3 + 32) + 40 * v11;
      if ( !*(_BYTE *)v12 )
      {
        v13 = *(_DWORD *)(v12 + 8);
        if ( v13 )
        {
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            break;
        }
      }
LABEL_15:
      if ( v10 == ++v11 )
        return result;
    }
    v14 = *(_QWORD *)(v3 + 16);
    v15 = v13;
    if ( *(unsigned __int16 *)(v14 + 2) > (unsigned int)v11 )
    {
      v60 = v10;
      v49 = sub_1F3AD60(v4[3], v14, v11, (_QWORD *)v4[4], v4[1]);
      v16 = (__int64 *)(v4[9] + 8LL * v13);
      v15 = v13;
      v10 = v60;
      if ( *v16 )
      {
        if ( *v16 == v49 && v49 )
          goto LABEL_22;
      }
      else if ( v49 )
      {
        *v16 = v49;
        goto LABEL_22;
      }
    }
    else
    {
      v16 = (__int64 *)(v4[9] + 8LL * v13);
    }
    *v16 = -1;
LABEL_22:
    v62 = v10;
    v64 = v15;
    v66 = v13;
    v67 = v12;
    sub_20E64D0((__int64)(v4 + 12), &v66);
    v17 = (_QWORD *)v4[4];
    if ( !v17 )
      BUG();
    v18 = v17[1];
    v19 = 0;
    v20 = v17[7];
    v10 = v62;
    v21 = *(_DWORD *)(v18 + 24 * v64 + 16);
    v22 = (v21 & 0xF) * v13;
    result = 0;
    v23 = (_WORD *)(v20 + 2LL * (v21 >> 4));
    v24 = v23 + 1;
    v25 = *v23 + v22;
LABEL_24:
    v26 = v24;
    while ( 1 )
    {
      v27 = v26;
      if ( !v26 )
      {
        v29 = v19;
        v30 = 0;
        goto LABEL_28;
      }
      v28 = (unsigned __int16 *)(v17[6] + 4LL * v25);
      v29 = *v28;
      result = v28[1];
      if ( v29 )
        break;
LABEL_70:
      v57 = *v26;
      v24 = 0;
      ++v26;
      if ( !v57 )
        goto LABEL_24;
      v25 += v57;
    }
    while ( 1 )
    {
      v30 = (__int16 *)(v20 + 2LL * *(unsigned int *)(v18 + 24LL * v29 + 8));
      if ( v30 )
        break;
      if ( !(_WORD)result )
      {
        v19 = v29;
        goto LABEL_70;
      }
      v29 = result;
      result = 0;
    }
LABEL_28:
    while ( v27 )
    {
      while ( 1 )
      {
        v31 = (_DWORD *)(v4[18] + 4LL * v29);
        if ( *v31 == -1 )
        {
          *v31 = a3;
          *(_DWORD *)(v4[21] + 4LL * v29) = -1;
        }
        v32 = *v30++;
        if ( !v32 )
          break;
        v29 += v32;
      }
      if ( (_WORD)result )
      {
        v48 = (unsigned __int16)result;
        v29 = result;
        result = 0;
        v30 = (__int16 *)(v17[7] + 2LL * *(unsigned int *)(v17[1] + 24 * v48 + 8));
      }
      else
      {
        result = *v27;
        v25 += result;
        if ( (_WORD)result )
        {
          ++v27;
          v46 = (unsigned __int16 *)(v17[6] + 4LL * v25);
          v47 = *v46;
          result = v46[1];
          v29 = v47;
          v30 = (__int16 *)(v17[7] + 2LL * *(unsigned int *)(v17[1] + 24 * v47 + 8));
        }
        else
        {
          v30 = 0;
          v27 = 0;
        }
      }
    }
    goto LABEL_15;
  }
  return result;
}
