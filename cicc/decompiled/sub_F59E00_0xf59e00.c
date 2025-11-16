// Function: sub_F59E00
// Address: 0xf59e00
//
__int64 __fastcall sub_F59E00(__int64 a1, __int64 a2)
{
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // r13d
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 k; // rcx
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rsi
  bool v24; // zf
  _QWORD *v25; // rax
  _QWORD *i; // rdx
  unsigned __int8 v27; // r14
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  int v37; // esi
  __int64 *v38; // rdx
  int v39; // eax
  __int64 *v40; // rax
  unsigned __int8 v41; // al
  __int64 *v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r9
  unsigned __int8 v45; // r8
  __int64 *v46; // rax
  unsigned int v47; // ecx
  _QWORD *v48; // rdi
  __int64 v49; // r9
  unsigned int v50; // eax
  int v51; // eax
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  int v54; // r15d
  __int64 v55; // r13
  _QWORD *v56; // rax
  _QWORD *j; // rdx
  _QWORD *v58; // r9
  unsigned __int8 v59; // [rsp+7h] [rbp-79h]
  __int64 v60; // [rsp+8h] [rbp-78h]
  __int64 v61; // [rsp+8h] [rbp-78h]
  unsigned __int8 v62; // [rsp+8h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-68h] BYREF
  __int64 *v64; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v65; // [rsp+28h] [rbp-58h] BYREF
  __int64 v66; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v67; // [rsp+38h] [rbp-48h]
  __int64 v68; // [rsp+40h] [rbp-40h]
  __int64 v69; // [rsp+48h] [rbp-38h]

  v4 = qword_4F8C008;
  v5 = sub_AA5930(a1);
  v7 = v4 + 1;
  if ( !v7 )
  {
LABEL_23:
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v66 = 1;
    if ( 4 * (_DWORD)qword_4F8C008 )
    {
      v19 = (16 * (int)qword_4F8C008 / 3u + 1) | ((unsigned __int64)(16 * (int)qword_4F8C008 / 3u + 1) >> 1);
      v20 = (((v19 >> 2) | v19) >> 4) | (v19 >> 2) | v19;
      sub_F59C70((__int64)&v66, ((((v20 >> 8) | v20) >> 16) | (v20 >> 8) | v20) + 1);
    }
    v21 = *(_QWORD *)(a1 + 56);
    v59 = 0;
    if ( *(_BYTE *)(v21 - 24) != 84 )
    {
LABEL_21:
      sub_C7D6A0((__int64)v67, 8LL * (unsigned int)v69, 8);
      return v59;
    }
    v22 = *(_QWORD *)(v21 + 8);
    v23 = v21 - 24;
    v24 = *(_BYTE *)(a2 + 28) == 0;
    v63 = v23;
    if ( v24 )
      goto LABEL_33;
LABEL_27:
    v25 = *(_QWORD **)(a2 + 8);
    for ( i = &v25[*(unsigned int *)(a2 + 20)]; i != v25; ++v25 )
    {
      if ( *v25 == v23 )
        goto LABEL_31;
    }
LABEL_34:
    v27 = sub_F59AD0((__int64)&v66, &v63, &v64);
    if ( v27 )
    {
      sub_BD84D0(v63, *v64);
      v32 = v63;
      if ( *(_BYTE *)(a2 + 28) )
      {
        v33 = *(__int64 **)(a2 + 8);
        v29 = *(unsigned int *)(a2 + 20);
        v28 = &v33[v29];
        if ( v33 == v28 )
        {
LABEL_73:
          if ( (unsigned int)v29 >= *(_DWORD *)(a2 + 16) )
            goto LABEL_74;
          *(_DWORD *)(a2 + 20) = v29 + 1;
          *v28 = v32;
          ++*(_QWORD *)a2;
        }
        else
        {
          while ( v63 != *v33 )
          {
            if ( v28 == ++v33 )
              goto LABEL_73;
          }
        }
LABEL_40:
        ++v66;
        if ( (_DWORD)v68 )
        {
          v47 = 4 * v68;
          v34 = (unsigned int)v69;
          if ( (unsigned int)(4 * v68) < 0x40 )
            v47 = 64;
          if ( v47 >= (unsigned int)v69 )
            goto LABEL_43;
          v48 = v67;
          v49 = (unsigned int)v69;
          if ( (_DWORD)v68 == 1 )
          {
            v55 = 1024;
            v54 = 128;
          }
          else
          {
            _BitScanReverse(&v50, v68 - 1);
            v51 = 1 << (33 - (v50 ^ 0x1F));
            if ( v51 < 64 )
              v51 = 64;
            if ( v51 == (_DWORD)v69 )
            {
              v68 = 0;
              v58 = &v67[v49];
              do
              {
                if ( v48 )
                  *v48 = -4096;
                ++v48;
              }
              while ( v58 != v48 );
              goto LABEL_46;
            }
            v52 = (4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1);
            v53 = ((v52 | (v52 >> 2)) >> 4) | v52 | (v52 >> 2) | ((((v52 | (v52 >> 2)) >> 4) | v52 | (v52 >> 2)) >> 8);
            v54 = (v53 | (v53 >> 16)) + 1;
            v55 = 8 * ((v53 | (v53 >> 16)) + 1);
          }
          sub_C7D6A0((__int64)v67, v49 * 8, 8);
          LODWORD(v69) = v54;
          v56 = (_QWORD *)sub_C7D670(v55, 8);
          v68 = 0;
          v67 = v56;
          for ( j = &v56[(unsigned int)v69]; j != v56; ++v56 )
          {
            if ( v56 )
              *v56 = -4096;
          }
        }
        else if ( HIDWORD(v68) )
        {
          v34 = (unsigned int)v69;
          if ( (unsigned int)v69 <= 0x40 )
          {
LABEL_43:
            v35 = v67;
            v36 = &v67[v34];
            if ( v67 != v36 )
            {
              do
                *v35++ = -4096;
              while ( v36 != v35 );
            }
            v68 = 0;
            goto LABEL_46;
          }
          sub_C7D6A0((__int64)v67, 8LL * (unsigned int)v69, 8);
          v67 = 0;
          v68 = 0;
          LODWORD(v69) = 0;
        }
LABEL_46:
        v59 = v27;
        v22 = *(_QWORD *)(a1 + 56);
        goto LABEL_31;
      }
LABEL_74:
      sub_C8CC70(a2, v63, (__int64)v28, v29, v30, v31);
      goto LABEL_40;
    }
    v37 = v69;
    v38 = v64;
    ++v66;
    v39 = v68 + 1;
    v65 = v64;
    if ( 4 * ((int)v68 + 1) >= (unsigned int)(3 * v69) )
    {
      v37 = 2 * v69;
    }
    else if ( (int)v69 - HIDWORD(v68) - v39 > (unsigned int)v69 >> 3 )
    {
      goto LABEL_49;
    }
    sub_F59C70((__int64)&v66, v37);
    sub_F59AD0((__int64)&v66, &v63, &v65);
    v38 = v65;
    v39 = v68 + 1;
LABEL_49:
    LODWORD(v68) = v39;
    if ( *v38 != -4096 )
      --HIDWORD(v68);
    *v38 = v63;
LABEL_31:
    while ( 1 )
    {
      v23 = v22 - 24;
      if ( *(_BYTE *)(v22 - 24) != 84 )
        goto LABEL_21;
      v24 = *(_BYTE *)(a2 + 28) == 0;
      v22 = *(_QWORD *)(v22 + 8);
      v63 = v23;
      if ( !v24 )
        goto LABEL_27;
LABEL_33:
      if ( !sub_C8CA60(a2, v23) )
        goto LABEL_34;
    }
  }
  v8 = v6;
  v9 = v5;
  while ( v8 != v9 )
  {
    v7 -= (unsigned __int8)sub_F4EE10(v9);
    if ( !v10 )
      BUG();
    v11 = *(_QWORD *)(v10 + 32);
    if ( !v11 )
      BUG();
    v9 = 0;
    if ( *(_BYTE *)(v11 - 24) == 84 )
      v9 = v11 - 24;
    if ( !v7 )
      goto LABEL_23;
  }
  v59 = 0;
  v12 = *(_QWORD *)(a1 + 56);
LABEL_10:
  if ( !v12 )
    BUG();
  if ( *(_BYTE *)(v12 - 24) == 84 )
  {
    v13 = v12 - 24;
    v12 = *(_QWORD *)(v12 + 8);
    for ( k = v12; ; k = *(_QWORD *)(v61 + 8) )
    {
      while ( 1 )
      {
        if ( !k )
          BUG();
        if ( *(_BYTE *)(k - 24) != 84 )
          goto LABEL_10;
        v15 = k - 24;
        if ( *(_BYTE *)(a2 + 28) )
          break;
        v60 = k;
        v40 = sub_C8CA60(a2, k - 24);
        k = v60;
        if ( !v40 )
          goto LABEL_53;
LABEL_20:
        k = *(_QWORD *)(k + 8);
      }
      v16 = *(_QWORD **)(a2 + 8);
      v17 = &v16[*(unsigned int *)(a2 + 20)];
      if ( v16 != v17 )
      {
        while ( v15 != *v16 )
        {
          if ( v17 == ++v16 )
            goto LABEL_53;
        }
        goto LABEL_20;
      }
LABEL_53:
      v61 = k;
      v41 = sub_B46130(v15, v13, 0);
      if ( v41 )
      {
        v62 = v41;
        sub_BD84D0(v15, v13);
        v45 = v62;
        if ( !*(_BYTE *)(a2 + 28) )
          goto LABEL_79;
        v46 = *(__int64 **)(a2 + 8);
        v43 = *(unsigned int *)(a2 + 20);
        v42 = &v46[v43];
        if ( v46 == v42 )
        {
LABEL_78:
          if ( (unsigned int)v43 < *(_DWORD *)(a2 + 16) )
          {
            *(_DWORD *)(a2 + 20) = v43 + 1;
            *v42 = v15;
            ++*(_QWORD *)a2;
          }
          else
          {
LABEL_79:
            sub_C8CC70(a2, v15, (__int64)v42, v43, v62, v44);
            v45 = v62;
          }
        }
        else
        {
          while ( v15 != *v46 )
          {
            if ( v42 == ++v46 )
              goto LABEL_78;
          }
        }
        v59 = v45;
        v12 = *(_QWORD *)(a1 + 56);
        goto LABEL_10;
      }
    }
  }
  return v59;
}
