// Function: sub_1DF3D00
// Address: 0x1df3d00
//
__int64 __fastcall sub_1DF3D00(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // r8
  int v3; // r9d
  int v5; // r12d
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned int v8; // ecx
  __int64 result; // rax
  _WORD *v10; // rcx
  unsigned __int16 *v11; // r10
  unsigned __int16 v12; // r15
  unsigned __int16 *v13; // rsi
  unsigned __int16 *v14; // r14
  unsigned __int16 *v15; // rax
  int v16; // ecx
  unsigned __int16 *v17; // rbx
  unsigned __int16 *v18; // r9
  int v19; // esi
  int v20; // esi
  __int64 v21; // r10
  unsigned int v22; // edi
  int *v23; // rdx
  int v24; // ebx
  int v25; // esi
  int v26; // esi
  __int64 v27; // r10
  unsigned int v28; // edi
  int *v29; // rdx
  int v30; // ebx
  __int64 v31; // rdx
  __int64 v32; // rdi
  unsigned int v33; // esi
  int *v34; // rbx
  int v35; // r10d
  unsigned __int64 v36; // rdi
  unsigned __int16 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // edx
  int v41; // r11d
  int v42; // ebx
  int v43; // edx
  int v44; // r11d
  int v45; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int16 *v46; // [rsp-48h] [rbp-48h]
  _QWORD *v47; // [rsp-40h] [rbp-40h]
  int v48; // [rsp-40h] [rbp-40h]

  v2 = *(_QWORD **)(a1 + 232);
  if ( !v2 )
    BUG();
  v3 = 0;
  v5 = 0;
  v6 = v2[1];
  v7 = v2[7];
  v8 = *(_DWORD *)(v6 + 24LL * a2 + 16);
  result = v8 & 0xF;
  v10 = (_WORD *)(v7 + 2LL * (v8 >> 4));
  v11 = v10 + 1;
  v12 = *v10 + a2 * result;
LABEL_3:
  v13 = v11;
  while ( 1 )
  {
    v14 = v13;
    if ( !v13 )
    {
      v16 = v3;
      v17 = 0;
      goto LABEL_7;
    }
    v15 = (unsigned __int16 *)(v2[6] + 4LL * v12);
    v16 = *v15;
    v5 = v15[1];
    if ( (_WORD)v16 )
      break;
LABEL_41:
    result = *v13;
    v11 = 0;
    ++v13;
    if ( !(_WORD)result )
      goto LABEL_3;
    v12 += result;
  }
  while ( 1 )
  {
    result = *(unsigned int *)(v6 + 24LL * (unsigned __int16)v16 + 8);
    v17 = (unsigned __int16 *)(v7 + 2 * result);
    if ( v17 )
      break;
    if ( !(_WORD)v5 )
    {
      v3 = v16;
      goto LABEL_41;
    }
    v16 = v5;
    v5 = 0;
  }
LABEL_7:
  v18 = v17;
  while ( v14 )
  {
    while ( 1 )
    {
      v19 = *(_DWORD *)(a1 + 472);
      if ( v19 )
      {
        v20 = v19 - 1;
        v21 = *(_QWORD *)(a1 + 456);
        v22 = v20 & (37 * (unsigned __int16)v16);
        v23 = (int *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( (unsigned __int16)v16 == *v23 )
        {
LABEL_11:
          *v23 = -2;
          --*(_DWORD *)(a1 + 464);
          ++*(_DWORD *)(a1 + 468);
        }
        else
        {
          v40 = 1;
          while ( v24 != -1 )
          {
            v41 = v40 + 1;
            v22 = v20 & (v40 + v22);
            v23 = (int *)(v21 + 16LL * v22);
            v24 = *v23;
            if ( (unsigned __int16)v16 == *v23 )
              goto LABEL_11;
            v40 = v41;
          }
        }
      }
      v25 = *(_DWORD *)(a1 + 440);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 424);
        v28 = v26 & (37 * (unsigned __int16)v16);
        v29 = (int *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( (unsigned __int16)v16 == *v29 )
        {
LABEL_14:
          *v29 = -2;
          --*(_DWORD *)(a1 + 432);
          ++*(_DWORD *)(a1 + 436);
        }
        else
        {
          v43 = 1;
          while ( v30 != -1 )
          {
            v44 = v43 + 1;
            v28 = v26 & (v43 + v28);
            v29 = (int *)(v27 + 16LL * v28);
            v30 = *v29;
            if ( (unsigned __int16)v16 == *v29 )
              goto LABEL_14;
            v43 = v44;
          }
        }
      }
      v31 = *(unsigned int *)(a1 + 504);
      if ( (_DWORD)v31 )
      {
        v32 = *(_QWORD *)(a1 + 488);
        v33 = (v31 - 1) & (37 * (unsigned __int16)v16);
        v34 = (int *)(v32 + 40LL * v33);
        v35 = *v34;
        if ( (unsigned __int16)v16 == *v34 )
        {
LABEL_17:
          if ( v34 != (int *)(v32 + 40 * v31) )
          {
            v45 = v16;
            v46 = v18;
            v47 = v2;
            sub_1DF39E0(a1 + 416, (unsigned int **)v34 + 1, *(_QWORD *)(a1 + 232));
            v36 = *((_QWORD *)v34 + 1);
            v16 = v45;
            v2 = v47;
            v18 = v46;
            if ( (int *)v36 != v34 + 6 )
            {
              _libc_free(v36);
              v16 = v45;
              v18 = v46;
              v2 = v47;
            }
            *v34 = -2;
            --*(_DWORD *)(a1 + 496);
            ++*(_DWORD *)(a1 + 500);
          }
        }
        else
        {
          v42 = 1;
          while ( v35 != -1 )
          {
            v33 = (v31 - 1) & (v42 + v33);
            v48 = v42 + 1;
            v34 = (int *)(v32 + 40LL * v33);
            v35 = *v34;
            if ( (unsigned __int16)v16 == *v34 )
              goto LABEL_17;
            v42 = v48;
          }
        }
      }
      result = *v18++;
      v16 += result;
      if ( (_WORD)result )
        break;
      if ( (_WORD)v5 )
      {
        v38 = v2[1] + 24LL * (unsigned __int16)v5;
        v16 = v5;
        v5 = 0;
        v39 = *(unsigned int *)(v38 + 8);
        result = v2[7];
        v17 = (unsigned __int16 *)(result + 2 * v39);
        goto LABEL_7;
      }
      v5 = *v14;
      v12 += v5;
      if ( !(_WORD)v5 )
      {
        v17 = 0;
        v14 = 0;
        goto LABEL_7;
      }
      ++v14;
      v37 = (unsigned __int16 *)(v2[6] + 4LL * v12);
      v16 = *v37;
      v5 = v37[1];
      result = v2[7];
      v18 = (unsigned __int16 *)(result + 2LL * *(unsigned int *)(v2[1] + 24LL * (unsigned __int16)v16 + 8));
      if ( !v14 )
        return result;
    }
  }
  return result;
}
