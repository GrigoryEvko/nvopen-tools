// Function: sub_3871F10
// Address: 0x3871f10
//
__int64 __fastcall sub_3871F10(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // r12
  __int64 v5; // rbx
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // r14d
  int v12; // r11d
  __int64 v13; // r8
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // rsi
  _QWORD *v17; // r10
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rcx
  _QWORD *v21; // rcx
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // r14
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rax
  _BYTE *v30; // r15
  _BYTE *v32; // r13
  unsigned int v33; // r14d
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rdx
  _QWORD *v38; // r12
  __int64 v39; // r15
  __int64 v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // r14
  int v43; // r13d
  __int64 v44; // r12
  __int64 v45; // r15
  __int64 v46; // rbx
  _QWORD *v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // rsi
  unsigned int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // rdi
  _QWORD *v53; // rsi
  int v54; // eax
  int v55; // edx
  int v56; // eax
  int v57; // edx
  int v58; // esi
  int v59; // edi
  __int64 v60; // [rsp+0h] [rbp-90h]
  __int64 v61; // [rsp+8h] [rbp-88h]
  __int64 v62; // [rsp+10h] [rbp-80h]
  _QWORD *v63; // [rsp+10h] [rbp-80h]
  __int64 v64; // [rsp+18h] [rbp-78h]
  _QWORD *v65; // [rsp+18h] [rbp-78h]
  _QWORD *v66; // [rsp+20h] [rbp-70h]
  unsigned __int8 v67; // [rsp+20h] [rbp-70h]
  __int64 v68; // [rsp+28h] [rbp-68h]
  _BYTE *v69; // [rsp+30h] [rbp-60h] BYREF
  __int64 v70; // [rsp+38h] [rbp-58h]
  _BYTE v71[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = a2;
  v4 = a1;
  v5 = a3;
  LOBYTE(v6) = sub_15CCEE0(*(_QWORD *)(*a1 + 56LL), a2, a3);
  v7 = v6;
  if ( (_BYTE)v6
    || *(_BYTE *)(v5 + 16) == 77
    || !sub_15CC8F0(*(_QWORD *)(*a1 + 56LL), *(_QWORD *)(v5 + 40), *(_QWORD *)(a2 + 40)) )
  {
    return v7;
  }
  v8 = *(_QWORD *)(v5 + 40);
  v9 = *(_QWORD *)(a2 + 40);
  if ( v8 == v9 )
    goto LABEL_20;
  v10 = *(_QWORD *)(*a1 + 64LL);
  v11 = *(_DWORD *)(v10 + 24);
  v68 = v10;
  if ( !v11 )
    goto LABEL_20;
  v12 = v11 - 1;
  v13 = *(_QWORD *)(v10 + 8);
  v14 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( *v15 == v9 )
  {
LABEL_7:
    v17 = (_QWORD *)v15[1];
  }
  else
  {
    v57 = 1;
    while ( v16 != -8 )
    {
      v59 = v57 + 1;
      v14 = v12 & (v57 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v9 == *v15 )
        goto LABEL_7;
      v57 = v59;
    }
    v17 = 0;
  }
  v18 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v19 = (__int64 *)(v13 + 16LL * v18);
  v20 = *v19;
  if ( v8 == *v19 )
  {
LABEL_9:
    v21 = (_QWORD *)v19[1];
    if ( v21 == v17 )
      goto LABEL_20;
    if ( v21 )
    {
      if ( v21 != v17 )
      {
        if ( v17 )
        {
          v22 = v17;
          while ( 1 )
          {
            v22 = (_QWORD *)*v22;
            if ( v21 == v22 )
              goto LABEL_16;
            if ( !v22 )
            {
              v42 = *(_QWORD *)(v3 + 8);
              if ( !v42 )
                goto LABEL_16;
              goto LABEL_44;
            }
          }
        }
        v42 = *(_QWORD *)(v3 + 8);
        if ( !v42 )
        {
LABEL_20:
          v69 = v71;
          v70 = 0x400000000LL;
          do
          {
            v26 = v3;
            v3 = sub_3870570((__int64)v4, (__int64 *)v3, v5, 1);
            if ( !v3 )
            {
              v30 = v69;
              goto LABEL_26;
            }
            v29 = (unsigned int)v70;
            if ( (unsigned int)v70 >= HIDWORD(v70) )
            {
              sub_16CD150((__int64)&v69, v71, 0, 8, v27, v28);
              v29 = (unsigned int)v70;
            }
            *(_QWORD *)&v69[8 * v29] = v26;
            v24 = *v4;
            LODWORD(v70) = v70 + 1;
            LOBYTE(v25) = sub_15CCEE0(*(_QWORD *)(v24 + 56), v3, v5);
          }
          while ( !(_BYTE)v25 );
          v32 = v69;
          v33 = v25;
          v30 = &v69[8 * (unsigned int)v70];
          if ( v69 == v30 )
          {
            v7 = v25;
          }
          else
          {
            do
            {
              v34 = *((_QWORD *)v30 - 1);
              v30 -= 8;
              sub_38707D0((__int64)v4, v34);
              sub_15F22F0(*(_QWORD **)v30, v5);
            }
            while ( v32 != v30 );
            v30 = v69;
            v7 = v33;
          }
LABEL_26:
          if ( v30 != v71 )
            _libc_free((unsigned __int64)v30);
          return v7;
        }
LABEL_44:
        v67 = v7;
        v43 = v12;
        v63 = v4;
        v44 = *(_QWORD *)(v5 + 40);
        v61 = v3;
        v45 = v13;
        v60 = v5;
        v46 = v42;
        v47 = v21;
        v65 = v17;
        do
        {
          v48 = sub_1648700(v46);
          if ( *((_BYTE *)v48 + 16) == 77 )
          {
            if ( (*((_BYTE *)v48 + 23) & 0x40) != 0 )
              v53 = (_QWORD *)*(v48 - 1);
            else
              v53 = &v48[-3 * (*((_DWORD *)v48 + 5) & 0xFFFFFFF)];
            v49 = v53[3 * *((unsigned int *)v48 + 14) + 1 + -1431655765 * (unsigned int)((v46 - (__int64)v53) >> 3)];
          }
          else
          {
            v49 = v48[5];
          }
          if ( v44 != v49 )
          {
            v50 = v43 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
            v51 = (__int64 *)(v45 + 16LL * v50);
            v52 = *v51;
            if ( *v51 != v49 )
            {
              v54 = 1;
              while ( v52 != -8 )
              {
                v55 = v54 + 1;
                v50 = v43 & (v54 + v50);
                v51 = (__int64 *)(v45 + 16LL * v50);
                v52 = *v51;
                if ( *v51 == v49 )
                  goto LABEL_50;
                v54 = v55;
              }
              return v67;
            }
LABEL_50:
            if ( v47 != (_QWORD *)v51[1] )
              return v67;
          }
          v46 = *(_QWORD *)(v46 + 8);
        }
        while ( v46 );
        v8 = v44;
        v17 = v65;
        v7 = v67;
        v21 = v47;
        v4 = v63;
        v3 = v61;
        v5 = v60;
      }
      if ( v17 )
      {
LABEL_16:
        if ( v17 != v21 )
        {
          v23 = v21;
          while ( 1 )
          {
            v23 = (_QWORD *)*v23;
            if ( v17 == v23 )
              break;
            if ( !v23 )
              goto LABEL_33;
          }
        }
        goto LABEL_20;
      }
      goto LABEL_20;
    }
    if ( !v17 )
      goto LABEL_20;
  }
  else
  {
    v56 = 1;
    while ( v20 != -8 )
    {
      v58 = v56 + 1;
      v18 = v12 & (v56 + v18);
      v19 = (__int64 *)(v13 + 16LL * v18);
      v20 = *v19;
      if ( v8 == *v19 )
        goto LABEL_9;
      v56 = v58;
    }
    if ( !v17 )
      goto LABEL_20;
    v21 = 0;
  }
LABEL_33:
  if ( *(_BYTE *)(v3 + 16) == 77 )
    return v7;
  v35 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
  {
    v36 = *(_QWORD *)(v3 - 8);
    v37 = v36 + v35;
  }
  else
  {
    v37 = v3;
    v36 = v3 - v35;
  }
  if ( v37 == v36 )
    goto LABEL_20;
  v66 = v4;
  v38 = v21;
  v64 = v3;
  v39 = v37;
  v62 = v5;
  v40 = v8;
  while ( *(_BYTE *)(*(_QWORD *)v36 + 16LL) > 0x17u )
  {
    v41 = *(_QWORD *)(*(_QWORD *)v36 + 40LL);
    if ( v40 != v41 && v38 != (_QWORD *)sub_13AE450(v68, v41) )
      break;
    v36 += 24;
    if ( v39 == v36 )
    {
      v4 = v66;
      v3 = v64;
      v5 = v62;
      goto LABEL_20;
    }
  }
  return v7;
}
