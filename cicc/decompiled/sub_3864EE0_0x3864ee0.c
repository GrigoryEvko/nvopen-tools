// Function: sub_3864EE0
// Address: 0x3864ee0
//
__int64 __fastcall sub_3864EE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        _DWORD *a9,
        int a10,
        char a11,
        char a12)
{
  unsigned __int64 v12; // r15
  __int64 v15; // r12
  __int64 *v16; // r14
  int v18; // r8d
  __int64 *v19; // rax
  __int64 v20; // r14
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // r14
  __int64 *v25; // rsi
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int64 v31; // rax
  __int64 v32; // r10
  unsigned int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // r9d
  __int64 *v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // rsi
  _QWORD *v39; // r8
  _QWORD *v40; // r9
  _QWORD *v41; // r10
  _BYTE *v42; // rdi
  _BYTE *v43; // rax
  int v44; // r12d
  int v45; // edx
  int v46; // edx
  _QWORD *v47; // [rsp+8h] [rbp-B8h]
  _QWORD *v48; // [rsp+18h] [rbp-A8h]
  __int64 *v49; // [rsp+20h] [rbp-A0h]
  _QWORD *v50; // [rsp+28h] [rbp-98h]
  __int64 v53; // [rsp+48h] [rbp-78h]
  unsigned __int64 v56; // [rsp+68h] [rbp-58h] BYREF
  _QWORD v57[10]; // [rsp+70h] [rbp-50h] BYREF

  v12 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = *(_QWORD *)(a1 + 416);
  v16 = sub_385D990(v15, a4, a3 & 0xFFFFFFFFFFFFFFF8LL, 0, a7, a8);
  if ( !sub_146CEE0(*(_QWORD *)(v15 + 112), (__int64)v16, a6) )
  {
    if ( *((_WORD *)v16 + 12) == 7 )
    {
      if ( v16[5] != 2 )
        return 0;
    }
    else
    {
      if ( !a12 )
        return 0;
      v19 = sub_14951F0(v15, v12, a7, a8);
      if ( !v19 || v19[5] != 2 )
        return 0;
    }
  }
  if ( a11 )
  {
    v20 = *(_QWORD *)(a1 + 416);
    v21 = sub_1494E70(v20, v12, a7, a8);
    if ( !sub_146CEE0(*(_QWORD *)(v20 + 112), (__int64)v21, a6)
      && sub_385E580(v20, v12, a6, a4, 0, 1, a7, a8) != 1
      && !sub_1494FF0(v20, v12, 1, a7, a8) )
    {
      v22 = sub_1494E70(*(_QWORD *)(a1 + 416), v12, a7, a8);
      if ( !a12 || *((_WORD *)v22 + 12) != 7 )
        return 0;
      sub_1497CC0(*(_QWORD *)(a1 + 416), v12, 1, a7, a8);
    }
  }
  if ( *(_DWORD *)(a1 + 80) )
  {
    v23 = *(_QWORD *)(a1 + 400);
    v57[1] = 1;
    v57[0] = v57;
    v57[2] = a3;
    v24 = *(__int64 **)(v23 + 16);
    v25 = (__int64 *)(v23 + 8);
    if ( !v24 )
      goto LABEL_65;
    v26 = (__int64 *)(v23 + 8);
    do
    {
      while ( 1 )
      {
        v27 = v24[2];
        v28 = v24[3];
        if ( a3 <= v24[6] )
          break;
        v24 = (__int64 *)v24[3];
        if ( !v28 )
          goto LABEL_24;
      }
      v26 = v24;
      v24 = (__int64 *)v24[2];
    }
    while ( v27 );
LABEL_24:
    if ( v25 == v26 || (v29 = v26[6], a3 < v29) )
LABEL_65:
      BUG();
    if ( (v26[5] & 1) == 0 )
    {
      v30 = v26[4];
      if ( (*(_BYTE *)(v30 + 8) & 1) != 0 )
      {
        v29 = *(_QWORD *)(v30 + 16);
      }
      else
      {
        v38 = *(_QWORD **)v30;
        if ( (*(_BYTE *)(*(_QWORD *)v30 + 8LL) & 1) == 0 )
        {
          v39 = (_QWORD *)*v38;
          if ( (*(_BYTE *)(*v38 + 8LL) & 1) != 0 )
          {
            v38 = (_QWORD *)*v38;
          }
          else
          {
            v40 = (_QWORD *)*v39;
            if ( (*(_BYTE *)(*v39 + 8LL) & 1) == 0 )
            {
              v41 = (_QWORD *)*v40;
              if ( (*(_BYTE *)(*v40 + 8LL) & 1) != 0 )
              {
                v40 = (_QWORD *)*v40;
              }
              else
              {
                v42 = (_BYTE *)*v41;
                v47 = (_QWORD *)*v40;
                if ( (*(_BYTE *)(*v41 + 8LL) & 1) == 0 )
                {
                  v48 = (_QWORD *)*v39;
                  v49 = v26;
                  v50 = (_QWORD *)*v38;
                  v53 = v26[4];
                  v43 = sub_3863620(v42);
                  v30 = v53;
                  v40 = v48;
                  v42 = v43;
                  *v47 = v43;
                  v26 = v49;
                  v39 = v50;
                }
                *v40 = v42;
                v40 = v42;
              }
              *v39 = v40;
            }
            *v38 = v40;
            v38 = v40;
          }
          *(_QWORD *)v30 = v38;
        }
        v26[4] = (__int64)v38;
        v29 = v38[2];
      }
    }
    v31 = v29 & 0xFFFFFFFFFFFFFFF8LL;
    v56 = v31;
    v32 = v31;
    v33 = *(_DWORD *)(a5 + 24);
    v34 = *(_QWORD *)(a5 + 8);
    if ( v33 )
    {
      v35 = (v33 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v36 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( v31 == *v36 )
      {
LABEL_31:
        v18 = *((_DWORD *)v36 + 2);
        if ( v18 )
          goto LABEL_8;
        v24 = v36;
LABEL_33:
        v18 = (*a9)++;
        *((_DWORD *)v24 + 2) = v18;
        goto LABEL_8;
      }
      v44 = 1;
      while ( v37 != -8 )
      {
        if ( v37 == -16 && !v24 )
          v24 = v36;
        v46 = v44++;
        v35 = (v33 - 1) & (v46 + v35);
        v36 = (__int64 *)(v34 + 16LL * v35);
        v37 = *v36;
        if ( v31 == *v36 )
          goto LABEL_31;
      }
      if ( !v24 )
        v24 = v36;
      ++*(_QWORD *)a5;
      v45 = *(_DWORD *)(a5 + 16) + 1;
      if ( 4 * v45 < 3 * v33 )
      {
        if ( v33 - *(_DWORD *)(a5 + 20) - v45 > v33 >> 3 )
        {
LABEL_51:
          *(_DWORD *)(a5 + 16) = v45;
          if ( *v24 != -8 )
            --*(_DWORD *)(a5 + 20);
          *v24 = v32;
          *((_DWORD *)v24 + 2) = 0;
          goto LABEL_33;
        }
LABEL_56:
        sub_177C7D0(a5, v33);
        sub_190E590(a5, (__int64 *)&v56, v57);
        v24 = (__int64 *)v57[0];
        v32 = v56;
        v45 = *(_DWORD *)(a5 + 16) + 1;
        goto LABEL_51;
      }
    }
    else
    {
      ++*(_QWORD *)a5;
    }
    v33 *= 2;
    goto LABEL_56;
  }
  v18 = (*a9)++;
LABEL_8:
  sub_3862E50(a2, a6, v12, (a3 >> 2) & 1, v18, a10, a7, a8, a4, *(_QWORD *)(a1 + 416));
  return 1;
}
