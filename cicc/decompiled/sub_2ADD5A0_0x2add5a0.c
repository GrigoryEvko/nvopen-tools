// Function: sub_2ADD5A0
// Address: 0x2add5a0
//
__int64 __fastcall sub_2ADD5A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v7; // r15d
  bool v8; // al
  __int64 v9; // r15
  unsigned __int16 v10; // r14
  _QWORD *v11; // rdi
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rdx
  int v22; // esi
  __int64 v23; // r9
  __int64 v24; // r10
  unsigned __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // r14
  unsigned __int8 *v31; // rax
  unsigned __int8 *v32; // r15
  unsigned __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r13
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rdx
  int v40; // r15d
  unsigned int v41; // edx
  __int64 v42; // rax
  unsigned int v43; // edx
  int v44; // eax
  __int64 v46; // r15
  _BYTE *v47; // rax
  unsigned int v48; // r15d
  __int64 v49; // rsi
  unsigned int v50; // r9d
  __int64 v51; // r15
  _QWORD *v52; // rdx
  _QWORD *v53; // r15
  __int64 v54; // r11
  __int64 v55; // r15
  unsigned __int64 v56; // rdi
  __int64 v57; // [rsp+0h] [rbp-80h]
  unsigned __int64 v58; // [rsp+8h] [rbp-78h]
  __int64 v59; // [rsp+8h] [rbp-78h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  unsigned int v61; // [rsp+10h] [rbp-70h]
  unsigned int v62; // [rsp+10h] [rbp-70h]
  unsigned int v63; // [rsp+18h] [rbp-68h]
  __int64 v64; // [rsp+18h] [rbp-68h]
  __int64 v65; // [rsp+18h] [rbp-68h]
  __int64 v66; // [rsp+20h] [rbp-60h]
  bool v67; // [rsp+20h] [rbp-60h]
  int v68; // [rsp+20h] [rbp-60h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  __int64 v70; // [rsp+20h] [rbp-60h]
  __int64 v71; // [rsp+20h] [rbp-60h]
  __int64 v73; // [rsp+30h] [rbp-50h]
  __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+30h] [rbp-50h]
  __int64 v76; // [rsp+30h] [rbp-50h]
  unsigned int v77; // [rsp+30h] [rbp-50h]
  __int64 v78; // [rsp+30h] [rbp-50h]
  __int64 v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+30h] [rbp-50h]
  __int64 v82; // [rsp+30h] [rbp-50h]
  __int64 v83; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int16 v84; // [rsp+48h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 416);
  v4 = *(_QWORD *)(a1 + 240);
  v5 = *(_QWORD *)(v3 + 8);
  if ( !v5 )
    return 0;
  if ( *(_BYTE *)v5 == 17 )
  {
    v7 = *(_DWORD *)(v5 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v5 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(v5 + 24);
    goto LABEL_5;
  }
  v46 = *(_QWORD *)(v5 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v46 + 8) - 17 <= 1 && *(_BYTE *)v5 <= 0x15u )
  {
    v47 = sub_AD7630(*(_QWORD *)(v3 + 8), 0, a3);
    if ( v47 && *v47 == 17 )
    {
      v48 = *((_DWORD *)v47 + 8);
      if ( v48 <= 0x40 )
        v8 = *((_QWORD *)v47 + 3) == 0;
      else
        v8 = v48 == (unsigned int)sub_C444A0((__int64)(v47 + 24));
LABEL_5:
      if ( !v8 )
        goto LABEL_6;
      return 0;
    }
    if ( *(_BYTE *)(v46 + 8) == 17 )
    {
      v40 = *(_DWORD *)(v46 + 32);
      if ( v40 )
      {
        v67 = 0;
        v41 = 0;
        while ( 1 )
        {
          v77 = v41;
          v42 = sub_AD69F0((unsigned __int8 *)v5, v41);
          if ( !v42 )
            break;
          v43 = v77;
          if ( *(_BYTE *)v42 != 13 )
          {
            if ( *(_BYTE *)v42 != 17 )
              break;
            if ( *(_DWORD *)(v42 + 32) <= 0x40u )
            {
              v67 = *(_QWORD *)(v42 + 24) == 0;
            }
            else
            {
              v68 = *(_DWORD *)(v42 + 32);
              v44 = sub_C444A0(v42 + 24);
              v43 = v77;
              v67 = v68 == v44;
            }
            if ( !v67 )
              break;
          }
          v41 = v43 + 1;
          if ( v40 == v41 )
          {
            if ( !v67 )
              break;
            return 0;
          }
        }
      }
    }
  }
LABEL_6:
  v9 = sub_AA54C0(v4);
  sub_B43C20((__int64)&v83, *(_QWORD *)v3);
  v10 = v84;
  v73 = v83;
  v11 = sub_BD2C40(72, 1u);
  if ( v11 )
    sub_B4C8F0((__int64)v11, v4, 1u, v73, v10);
  v12 = *(__int64 **)(v3 + 1808);
  if ( v12 )
    sub_D4F330(v12, *(_QWORD *)v3, *(_QWORD *)(v3 + 40));
  v13 = (_QWORD *)sub_986580(*(_QWORD *)v3);
  sub_B43D60(v13);
  sub_AA4AC0(*(_QWORD *)v3, v4 + 24);
  v14 = (unsigned __int8 *)sub_986580(v9);
  sub_B47210(v14, v4, *(_QWORD *)v3);
  v15 = *(_QWORD *)(v3 + 32);
  if ( v9 )
  {
    v66 = *(_QWORD *)v3;
    v16 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
    if ( (unsigned int)(*(_DWORD *)(v9 + 44) + 1) < *(_DWORD *)(v15 + 32) )
      goto LABEL_12;
LABEL_69:
    *(_BYTE *)(v15 + 112) = 0;
    v78 = v15;
    v18 = sub_22077B0(0x50u);
    v20 = v78;
    v21 = v66;
    if ( !v18 )
    {
      v17 = 0;
      goto LABEL_16;
    }
    *(_QWORD *)v18 = v66;
    v17 = 0;
    v22 = 0;
    *(_QWORD *)(v18 + 8) = 0;
    goto LABEL_15;
  }
  v16 = 0;
  v66 = *(_QWORD *)v3;
  if ( !*(_DWORD *)(v15 + 32) )
    goto LABEL_69;
LABEL_12:
  v74 = *(_QWORD *)(v3 + 32);
  v17 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + 8 * v16);
  *(_BYTE *)(v15 + 112) = 0;
  v18 = sub_22077B0(0x50u);
  v20 = v74;
  v21 = v66;
  if ( v18 )
  {
    *(_QWORD *)v18 = v66;
    *(_QWORD *)(v18 + 8) = v17;
    if ( v17 )
      v22 = *(_DWORD *)(v17 + 16) + 1;
    else
      v22 = 0;
LABEL_15:
    *(_DWORD *)(v18 + 16) = v22;
    *(_QWORD *)(v18 + 24) = v18 + 40;
    *(_QWORD *)(v18 + 32) = 0x400000000LL;
    *(_QWORD *)(v18 + 72) = -1;
  }
LABEL_16:
  if ( v21 )
  {
    v23 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
    v24 = 8 * v23;
  }
  else
  {
    v24 = 0;
    LODWORD(v23) = 0;
  }
  v25 = *(unsigned int *)(v20 + 32);
  if ( (unsigned int)v25 <= (unsigned int)v23 )
  {
    v49 = *(_QWORD *)(v20 + 104);
    v50 = v23 + 1;
    if ( *(_DWORD *)(v49 + 88) >= v50 )
      v50 = *(_DWORD *)(v49 + 88);
    if ( v50 != v25 )
    {
      v51 = 8LL * v50;
      if ( v50 < v25 )
      {
        v26 = *(_QWORD *)(v20 + 24);
        v54 = v26 + 8 * v25;
        v55 = v26 + v51;
        if ( v54 == v55 )
          goto LABEL_66;
        do
        {
          v19 = *(_QWORD *)(v54 - 8);
          v54 -= 8;
          if ( v19 )
          {
            v56 = *(_QWORD *)(v19 + 24);
            if ( v56 != v19 + 40 )
            {
              v57 = v18;
              v58 = v19;
              v61 = v50;
              v64 = v24;
              v70 = v54;
              v81 = v20;
              _libc_free(v56);
              v18 = v57;
              v19 = v58;
              v50 = v61;
              v24 = v64;
              v54 = v70;
              v20 = v81;
            }
            v59 = v18;
            v62 = v50;
            v65 = v24;
            v71 = v54;
            v82 = v20;
            j_j___libc_free_0(v19);
            v18 = v59;
            v50 = v62;
            v24 = v65;
            v54 = v71;
            v20 = v82;
          }
        }
        while ( v55 != v54 );
      }
      else
      {
        if ( v50 > (unsigned __int64)*(unsigned int *)(v20 + 36) )
        {
          v60 = v18;
          v63 = v50;
          v69 = v24;
          v79 = v20;
          sub_B1B4E0(v20 + 24, v50);
          v20 = v79;
          v18 = v60;
          v50 = v63;
          v24 = v69;
          v25 = *(unsigned int *)(v79 + 32);
        }
        v26 = *(_QWORD *)(v20 + 24);
        v52 = (_QWORD *)(v26 + 8 * v25);
        v53 = (_QWORD *)(v26 + v51);
        if ( v52 == v53 )
          goto LABEL_66;
        do
        {
          if ( v52 )
            *v52 = 0;
          ++v52;
        }
        while ( v53 != v52 );
      }
      v26 = *(_QWORD *)(v20 + 24);
LABEL_66:
      *(_DWORD *)(v20 + 32) = v50;
      goto LABEL_20;
    }
  }
  v26 = *(_QWORD *)(v20 + 24);
LABEL_20:
  v27 = *(_QWORD *)(v26 + v24);
  *(_QWORD *)(v26 + v24) = v18;
  if ( v27 )
  {
    v28 = *(_QWORD *)(v27 + 24);
    if ( v28 != v27 + 40 )
    {
      v75 = v18;
      _libc_free(v28);
      v18 = v75;
    }
    v76 = v18;
    j_j___libc_free_0(v27);
    v18 = v76;
  }
  if ( v17 )
  {
    v29 = *(unsigned int *)(v17 + 32);
    if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 36) )
    {
      v80 = v18;
      sub_C8D5F0(v17 + 24, (const void *)(v17 + 40), v29 + 1, 8u, v19, v29 + 1);
      v29 = *(unsigned int *)(v17 + 32);
      v18 = v80;
    }
    *(_QWORD *)(*(_QWORD *)(v17 + 24) + 8 * v29) = v18;
    ++*(_DWORD *)(v17 + 32);
  }
  sub_B1AEF0(*(_QWORD *)(v3 + 32), v4, *(_QWORD *)v3);
  v30 = *(_QWORD *)(v3 + 8);
  v31 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v32 = v31;
  if ( v31 )
    sub_B4C9A0((__int64)v31, a2, v4, v30, 3u, 0, 0, 0);
  if ( *(_BYTE *)(v3 + 1801) )
    sub_BC8EC0((__int64)v32, dword_439F0C8, 2, 0);
  v33 = sub_986580(*(_QWORD *)v3);
  sub_F34910(v33, v32);
  v36 = *(_QWORD *)v3;
  *(_QWORD *)(v3 + 8) = 0;
  if ( v36 )
  {
    v37 = *(unsigned int *)(a1 + 272);
    v38 = *(unsigned int *)(a1 + 276);
    if ( v37 + 1 > v38 )
    {
      sub_C8D5F0(a1 + 264, (const void *)(a1 + 280), v37 + 1, 8u, v34, v35);
      v37 = *(unsigned int *)(a1 + 272);
    }
    v39 = *(_QWORD *)(a1 + 264);
    *(_QWORD *)(v39 + 8 * v37) = v36;
    ++*(_DWORD *)(a1 + 272);
    *(_BYTE *)(a1 + 392) = 1;
    sub_2AB95C0(a1, v36, v39, v38, v34, v35);
  }
  return v36;
}
