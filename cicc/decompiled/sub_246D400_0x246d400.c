// Function: sub_246D400
// Address: 0x246d400
//
__int64 __fastcall sub_246D400(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r10
  int v11; // r8d
  unsigned int *v12; // rcx
  unsigned int v13; // esi
  unsigned int *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // r9d
  int v18; // esi
  int v19; // esi
  _DWORD *v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  unsigned int *v25; // rdi
  unsigned int v26; // ecx
  unsigned int *v27; // rdx
  __int64 v28; // r8
  signed int v29; // r15d
  signed int v30; // eax
  unsigned int *v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // r8
  unsigned __int64 v34; // rsi
  unsigned int *v35; // rax
  unsigned int v36; // ecx
  unsigned int *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 *v44; // rax
  int v46; // edi
  __int64 v47; // rax
  int v48; // ecx
  int v49; // ecx
  __int64 v50; // [rsp-10h] [rbp-130h]
  __int64 v52; // [rsp+10h] [rbp-110h]
  int v53; // [rsp+10h] [rbp-110h]
  __int64 v54; // [rsp+10h] [rbp-110h]
  __int64 v55; // [rsp+18h] [rbp-108h] BYREF
  unsigned int *v56; // [rsp+20h] [rbp-100h] BYREF
  __int64 v57; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v58[4]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v59; // [rsp+50h] [rbp-D0h]
  unsigned int *v60; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v61; // [rsp+68h] [rbp-B8h]
  unsigned int v62; // [rsp+6Ch] [rbp-B4h]
  _BYTE v63[16]; // [rsp+70h] [rbp-B0h] BYREF
  __int16 v64; // [rsp+80h] [rbp-A0h]

  v3 = (__int64 *)a2;
  v4 = a1;
  v55 = a3;
  if ( !a3 )
  {
    v47 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
    v55 = sub_ACD640(v47, 0, 0);
  }
  sub_B33910(&v57, (__int64 *)a2);
  v5 = *(_QWORD *)(a1 + 8);
  if ( *(int *)(v5 + 4) > 1 )
  {
    v52 = a1 + 1040;
    if ( !*(_DWORD *)(a1 + 1056) )
    {
      v6 = *(_QWORD *)(a1 + 640);
      if ( v6 != v6 + 24LL * *(unsigned int *)(a1 + 648) )
      {
        v8 = *(_QWORD *)(a1 + 640);
        v9 = v6 + 24LL * *(unsigned int *)(a1 + 648);
        while ( 1 )
        {
          v16 = sub_B10CD0(*(_QWORD *)(v8 + 16) + 48LL);
          v17 = *(_DWORD *)(a1 + 1064);
          v58[0] = v16;
          if ( !v17 )
            break;
          v10 = *(_QWORD *)(a1 + 1048);
          v11 = 1;
          v12 = 0;
          v13 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v14 = (unsigned int *)(v10 + 16LL * v13);
          v15 = *(_QWORD *)v14;
          if ( v16 == *(_QWORD *)v14 )
          {
LABEL_8:
            v8 += 24;
            ++v14[2];
            if ( v9 == v8 )
              goto LABEL_16;
          }
          else
          {
            while ( v15 != -4096 )
            {
              if ( !v12 && v15 == -8192 )
                v12 = v14;
              v13 = (v17 - 1) & (v11 + v13);
              v14 = (unsigned int *)(v10 + 16LL * v13);
              v15 = *(_QWORD *)v14;
              if ( v16 == *(_QWORD *)v14 )
                goto LABEL_8;
              ++v11;
            }
            v46 = *(_DWORD *)(a1 + 1056);
            if ( !v12 )
              v12 = v14;
            ++*(_QWORD *)(a1 + 1040);
            v19 = v46 + 1;
            v60 = v12;
            if ( 4 * (v46 + 1) >= 3 * v17 )
              goto LABEL_11;
            if ( v17 - *(_DWORD *)(a1 + 1060) - v19 > v17 >> 3 )
              goto LABEL_13;
            v18 = v17;
LABEL_12:
            sub_246D220(v52, v18);
            sub_246CE20(v52, v58, &v60);
            v16 = v58[0];
            v19 = *(_DWORD *)(a1 + 1056) + 1;
            v12 = v60;
LABEL_13:
            *(_DWORD *)(a1 + 1056) = v19;
            if ( *(_QWORD *)v12 != -4096 )
              --*(_DWORD *)(a1 + 1060);
            *(_QWORD *)v12 = v16;
            v8 += 24;
            v20 = v12 + 2;
            *v20 = 0;
            *v20 = 1;
            if ( v9 == v8 )
            {
LABEL_16:
              v4 = a1;
              v3 = (__int64 *)a2;
              goto LABEL_17;
            }
          }
        }
        ++*(_QWORD *)(a1 + 1040);
        v60 = 0;
LABEL_11:
        v18 = 2 * v17;
        goto LABEL_12;
      }
    }
LABEL_17:
    v21 = sub_B10CD0((__int64)&v57);
    v22 = *(_DWORD *)(v4 + 1064);
    v58[0] = v21;
    if ( v22 )
    {
      v23 = *(_QWORD *)(v4 + 1048);
      v24 = 1;
      v25 = 0;
      v26 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v27 = (unsigned int *)(v23 + 16LL * v26);
      v28 = *(_QWORD *)v27;
      if ( v21 == *(_QWORD *)v27 )
      {
LABEL_19:
        v29 = v27[2];
        goto LABEL_20;
      }
      while ( v28 != -4096 )
      {
        if ( !v25 && v28 == -8192 )
          v25 = v27;
        v26 = (v22 - 1) & (v24 + v26);
        v27 = (unsigned int *)(v23 + 16LL * v26);
        v28 = *(_QWORD *)v27;
        if ( v21 == *(_QWORD *)v27 )
          goto LABEL_19;
        ++v24;
      }
      v49 = *(_DWORD *)(v4 + 1056);
      if ( !v25 )
        v25 = v27;
      ++*(_QWORD *)(v4 + 1040);
      v48 = v49 + 1;
      v60 = v25;
      if ( 4 * v48 < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(v4 + 1060) - v48 > v22 >> 3 )
          goto LABEL_74;
        goto LABEL_73;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 1040);
      v60 = 0;
    }
    v22 *= 2;
LABEL_73:
    sub_246D220(v52, v22);
    sub_246CE20(v52, v58, &v60);
    v21 = v58[0];
    v25 = v60;
    v48 = *(_DWORD *)(v4 + 1056) + 1;
LABEL_74:
    *(_DWORD *)(v4 + 1056) = v48;
    if ( *(_QWORD *)v25 != -4096 )
      --*(_DWORD *)(v4 + 1060);
    *(_QWORD *)v25 = v21;
    v29 = 0;
    v25[2] = 0;
LABEL_20:
    v30 = qword_4FE7968;
    if ( v57 )
    {
      v53 = qword_4FE7968;
      sub_B91220((__int64)&v57, v57);
      v30 = v53;
    }
    if ( v30 > v29 )
      goto LABEL_45;
    if ( !v55 )
      goto LABEL_45;
    if ( *(_BYTE *)v55 <= 0x1Cu )
      goto LABEL_45;
    v31 = *(unsigned int **)(v55 + 48);
    v56 = v31;
    if ( !v31 )
      goto LABEL_45;
    sub_B96E90((__int64)&v56, (__int64)v31, 1);
    if ( !v56 )
      goto LABEL_45;
    sub_B33910(&v60, v3);
    if ( v56 == v60 )
    {
      if ( !v60 )
      {
LABEL_45:
        v5 = *(_QWORD *)(v4 + 8);
        goto LABEL_46;
      }
      sub_B91220((__int64)&v60, (__int64)v60);
LABEL_43:
      if ( v56 )
        sub_B91220((__int64)&v56, (__int64)v56);
      goto LABEL_45;
    }
    if ( v60 )
      sub_B91220((__int64)&v60, (__int64)v60);
    v32 = v3[7];
    if ( v32 )
      v32 -= 24;
    sub_23D0AB0((__int64)&v60, v32, 0, 0, 0);
    v58[0] = (__int64)v56;
    if ( v56 && (sub_B96E90((__int64)v58, (__int64)v56, 1), (v33 = v58[0]) != 0) )
    {
      v34 = v61;
      v35 = v60;
      v36 = v61;
      v37 = &v60[4 * v61];
      if ( v60 != v37 )
      {
        while ( *v35 )
        {
          v35 += 4;
          if ( v37 == v35 )
            goto LABEL_67;
        }
        *((_QWORD *)v35 + 1) = v58[0];
LABEL_39:
        v34 = v33;
        sub_B91220((__int64)v58, v33);
LABEL_40:
        v38 = v55;
        v39 = *(_QWORD *)(v4 + 8);
        v57 = v55;
        if ( *(int *)(v39 + 4) > 1 )
        {
          v59 = 257;
          v34 = *(_QWORD *)(v39 + 360);
          v38 = sub_921880(&v60, v34, *(_QWORD *)(v39 + 368), (int)&v57, 1, (__int64)v58, 0);
        }
        v55 = v38;
        sub_F94A20(&v60, v34);
        goto LABEL_43;
      }
LABEL_67:
      if ( v61 >= (unsigned __int64)v62 )
      {
        v34 = v61 + 1LL;
        if ( v62 < v34 )
        {
          v34 = (unsigned __int64)v63;
          v54 = v58[0];
          sub_C8D5F0((__int64)&v60, v63, v61 + 1LL, 0x10u, v58[0], (__int64)v63);
          v33 = v54;
          v37 = &v60[4 * v61];
        }
        *(_QWORD *)v37 = 0;
        *((_QWORD *)v37 + 1) = v33;
        v33 = v58[0];
        ++v61;
      }
      else
      {
        if ( v37 )
        {
          *v37 = 0;
          *((_QWORD *)v37 + 1) = v33;
          v36 = v61;
          v33 = v58[0];
        }
        v61 = v36 + 1;
      }
    }
    else
    {
      v34 = 0;
      sub_93FB40((__int64)&v60, 0);
      v33 = v58[0];
    }
    if ( !v33 )
      goto LABEL_40;
    goto LABEL_39;
  }
  if ( v57 )
  {
    sub_B91220((__int64)&v57, v57);
    v5 = *(_QWORD *)(a1 + 8);
  }
LABEL_46:
  v40 = *(_QWORD *)(v5 + 168);
  v41 = *(_QWORD *)(v5 + 176);
  if ( *(_BYTE *)v5 || *(_DWORD *)(v5 + 4) )
  {
    v64 = 257;
    v42 = sub_921880((unsigned int **)v3, v40, v41, (int)&v55, 1, (__int64)&v60, 0);
  }
  else
  {
    v64 = 257;
    v42 = sub_921880((unsigned int **)v3, v40, v41, 0, 0, (__int64)&v60, 0);
  }
  v43 = v42;
  v44 = (__int64 *)sub_BD5C60(v42);
  *(_QWORD *)(v43 + 72) = sub_A7A090((__int64 *)(v43 + 72), v44, -1, 32);
  return v50;
}
