// Function: sub_186A8A0
// Address: 0x186a8a0
//
__int64 __fastcall sub_186A8A0(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v4; // r12
  __int64 *v5; // r14
  __int64 v6; // r15
  int v7; // r8d
  int v8; // r9d
  __int64 i; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rax
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // r12d
  unsigned __int64 **v18; // r15
  __int64 v19; // rsi
  __int64 v20; // r12
  unsigned __int64 **v21; // rsi
  char *v22; // rax
  unsigned __int64 *v23; // rcx
  unsigned __int64 **v24; // rdi
  unsigned __int64 **k; // r13
  _QWORD *v26; // rax
  __int64 v27; // r12
  __int64 v29; // rax
  unsigned __int64 **v30; // rdx
  _BYTE *v31; // rdi
  __int64 v32; // rdx
  unsigned __int64 **v33; // rax
  size_t v34; // rdx
  unsigned __int64 *v35; // r13
  unsigned __int64 *v36; // r14
  _QWORD *v37; // rax
  unsigned __int64 v38; // rsi
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // r15
  __int64 j; // rdx
  __int64 v44; // rdx
  _QWORD *v45; // rdi
  __int64 v46; // rdx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rdx
  void *base; // [rsp+20h] [rbp-150h] BYREF
  __int64 v52; // [rsp+28h] [rbp-148h]
  _BYTE v53[128]; // [rsp+30h] [rbp-140h] BYREF
  unsigned __int64 *v54; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-B8h]
  _BYTE v56[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v4 = a2[4];
  base = v53;
  v52 = 0x1000000000LL;
  v54 = (unsigned __int64 *)v56;
  v55 = 0x1000000000LL;
  if ( a2 + 2 == (_QWORD *)v4 )
  {
    v17 = 0;
  }
  else
  {
    do
    {
      v5 = *(__int64 **)(v4 + 40);
      v6 = *v5;
      if ( *v5 && !sub_15E4F60(*v5) && (!a3 || (unsigned __int8)sub_1560180(v6 + 112, 3)) )
      {
        sub_159D9E0(v6);
        if ( (unsigned __int8)sub_15E36F0(v6) )
        {
          if ( (*(_BYTE *)(v6 + 32) & 0xFu) - 7 > 1 && *(_QWORD *)(v6 + 48) )
          {
            v29 = (unsigned int)v55;
            if ( (unsigned int)v55 >= HIDWORD(v55) )
            {
              sub_16CD150((__int64)&v54, v56, 0, 8, v7, v8);
              v29 = (unsigned int)v55;
            }
            v54[v29] = v6;
            LODWORD(v55) = v55 + 1;
          }
          else
          {
            for ( i = v5[2]; i != v5[1]; i = v5[2] )
            {
              --*(_DWORD *)(*(_QWORD *)(i - 8) + 32LL);
              v10 = v5[2];
              v11 = (_QWORD *)(v10 - 32);
              v5[2] = v10 - 32;
              v12 = *(_QWORD *)(v10 - 16);
              if ( v12 != 0 && v12 != -8 && v12 != -16 )
                sub_1649B30(v11);
            }
            sub_1398490(a2[7], (__int64)v5);
            v15 = (unsigned int)v52;
            if ( (unsigned int)v52 >= HIDWORD(v52) )
            {
              sub_16CD150((__int64)&base, v53, 0, 8, v13, v14);
              v15 = (unsigned int)v52;
            }
            *((_QWORD *)base + v15) = v5;
            LODWORD(v52) = v52 + 1;
          }
        }
      }
      v4 = sub_220EEE0(v4);
    }
    while ( a2 + 2 != (_QWORD *)v4 );
    if ( !(_DWORD)v55 || (sub_1B28CA0(*a2, &v54), v35 = v54, v36 = &v54[(unsigned int)v55], v54 == v36) )
    {
      v16 = (unsigned int)v52;
    }
    else
    {
      do
      {
        v37 = (_QWORD *)a2[3];
        v38 = *v35;
        v39 = (_QWORD *)v4;
        if ( v37 )
        {
          do
          {
            while ( 1 )
            {
              v40 = v37[2];
              v41 = v37[3];
              if ( v37[4] >= v38 )
                break;
              v37 = (_QWORD *)v37[3];
              if ( !v41 )
                goto LABEL_56;
            }
            v39 = v37;
            v37 = (_QWORD *)v37[2];
          }
          while ( v40 );
LABEL_56:
          if ( (_QWORD *)v4 != v39 && v39[4] > v38 )
            v39 = (_QWORD *)v4;
        }
        v42 = v39[5];
        for ( j = *(_QWORD *)(v42 + 16); j != *(_QWORD *)(v42 + 8); j = *(_QWORD *)(v42 + 16) )
        {
          --*(_DWORD *)(*(_QWORD *)(j - 8) + 32LL);
          v44 = *(_QWORD *)(v42 + 16);
          v45 = (_QWORD *)(v44 - 32);
          *(_QWORD *)(v42 + 16) = v44 - 32;
          v46 = *(_QWORD *)(v44 - 16);
          if ( v46 != 0 && v46 != -8 && v46 != -16 )
            sub_1649B30(v45);
        }
        sub_1398490(a2[7], v42);
        v49 = (unsigned int)v52;
        if ( (unsigned int)v52 >= HIDWORD(v52) )
        {
          sub_16CD150((__int64)&base, v53, 0, 8, v47, v48);
          v49 = (unsigned int)v52;
        }
        ++v35;
        *((_QWORD *)base + v49) = v42;
        v16 = (unsigned int)(v52 + 1);
        LODWORD(v52) = v52 + 1;
      }
      while ( v36 != v35 );
    }
    v17 = 0;
    if ( (_DWORD)v16 )
    {
      v18 = (unsigned __int64 **)base;
      v19 = v16;
      v20 = 8;
      if ( v16 != 1 )
      {
        qsort(base, (v19 * 8) >> 3, 8u, (__compar_fn_t)sub_1869D00);
        v18 = (unsigned __int64 **)base;
        v19 = (unsigned int)v52;
        v20 = v19 * 8;
      }
      v21 = &v18[v19];
      if ( v21 != v18 )
      {
        v22 = (char *)v18;
        do
        {
          v24 = (unsigned __int64 **)v22;
          v22 += 8;
          if ( v21 == (unsigned __int64 **)v22 )
            goto LABEL_26;
          v23 = (unsigned __int64 *)*((_QWORD *)v22 - 1);
        }
        while ( v23 != *(unsigned __int64 **)v22 );
        if ( v21 == v24 )
        {
LABEL_26:
          v20 = (char *)v21 - (char *)v18;
          goto LABEL_27;
        }
        v30 = v24 + 2;
        if ( v21 == v24 + 2 )
        {
          v20 = v22 - (char *)v18;
        }
        else
        {
          while ( 1 )
          {
            if ( *v30 != v23 )
            {
              v24[1] = *v30;
              ++v24;
            }
            if ( v21 == ++v30 )
              break;
            v23 = *v24;
          }
          v31 = v24 + 1;
          v32 = (unsigned int)v52;
          v33 = &v18[v32];
          v34 = v32 * 8 - v20;
          v20 = &v31[v34] - (_BYTE *)v18;
          if ( v21 != v33 )
            memmove(v31, v21, v34);
        }
      }
LABEL_27:
      LODWORD(v52) = v20 >> 3;
      for ( k = &v18[(unsigned int)v52]; k != v18; ++v18 )
      {
        v26 = (_QWORD *)sub_13977A0(a2, *v18);
        v27 = (__int64)v26;
        if ( v26 )
        {
          sub_15E3C20(v26);
          sub_1648B90(v27);
        }
      }
      v17 = 1;
    }
    if ( v54 != (unsigned __int64 *)v56 )
      _libc_free((unsigned __int64)v54);
  }
  if ( base != v53 )
    _libc_free((unsigned __int64)base);
  return v17;
}
