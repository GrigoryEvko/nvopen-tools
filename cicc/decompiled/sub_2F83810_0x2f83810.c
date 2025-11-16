// Function: sub_2F83810
// Address: 0x2f83810
//
__int64 __fastcall sub_2F83810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  __int64 v8; // r15
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rbx
  __int64 v13; // r14
  unsigned __int8 *v14; // r12
  unsigned int v15; // r13d
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // r14d
  unsigned __int8 **v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int8 *v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r13
  int v30; // r13d
  __int64 v31; // rax
  unsigned __int8 *v32; // r15
  unsigned __int8 *v33; // r14
  __int64 v34; // r12
  int v35; // edx
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  _QWORD *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r13
  int v43; // r13d
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // [rsp+0h] [rbp-190h]
  __int64 v48; // [rsp+8h] [rbp-188h]
  __int64 v49; // [rsp+18h] [rbp-178h]
  unsigned __int8 *v50; // [rsp+20h] [rbp-170h]
  __int64 v51; // [rsp+28h] [rbp-168h]
  unsigned __int8 *v52; // [rsp+28h] [rbp-168h]
  __int64 v53; // [rsp+38h] [rbp-158h]
  _QWORD v56[2]; // [rsp+60h] [rbp-130h] BYREF
  _QWORD *v57; // [rsp+70h] [rbp-120h] BYREF
  __int64 v58; // [rsp+78h] [rbp-118h]
  _QWORD v59[8]; // [rsp+80h] [rbp-110h] BYREF
  __int64 v60; // [rsp+C0h] [rbp-D0h] BYREF
  unsigned __int8 **v61; // [rsp+C8h] [rbp-C8h]
  __int64 v62; // [rsp+D0h] [rbp-C0h]
  int v63; // [rsp+D8h] [rbp-B8h]
  char v64; // [rsp+DCh] [rbp-B4h]
  char v65; // [rsp+E0h] [rbp-B0h] BYREF

  v7 = v59;
  v8 = a1;
  v59[0] = a2;
  v60 = 0;
  v62 = 16;
  v63 = 0;
  v64 = 1;
  v57 = v59;
  v61 = (unsigned __int8 **)&v65;
  v58 = 0x800000001LL;
  v9 = 1;
  while ( 1 )
  {
    v10 = v9--;
    v11 = v7[v10 - 1];
    LODWORD(v58) = v9;
    v12 = *(__int64 **)(v11 + 16);
    v53 = v11;
    if ( v12 )
      break;
LABEL_12:
    if ( !v9 )
    {
      v21 = 1;
      goto LABEL_28;
    }
  }
  v13 = a2;
  while ( 1 )
  {
    v14 = (unsigned __int8 *)v12[3];
    v15 = *v14 - 29;
    if ( *v14 == 62 )
      break;
    if ( v15 > 0x21 )
    {
      if ( *v14 == 85 )
      {
        if ( sub_B46A10(v12[3]) )
          goto LABEL_10;
        v27 = *((_QWORD *)v14 - 4);
        v10 = *((_DWORD *)v14 + 1) & 0x7FFFFFF;
        if ( !v27 )
          goto LABEL_41;
        if ( *(_BYTE *)v27 )
          goto LABEL_41;
        if ( *(_QWORD *)(v27 + 24) != *((_QWORD *)v14 + 10) )
          goto LABEL_41;
        if ( (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
          goto LABEL_41;
        v36 = *(_DWORD *)(v27 + 36);
        v11 = (unsigned int)(v36 - 238);
        if ( (unsigned int)v11 > 7 || ((1LL << ((unsigned __int8)v36 + 18)) & 0xAD) == 0 )
          goto LABEL_41;
        v37 = *v12;
        if ( (v36 == 238 || (unsigned int)(v36 - 240) <= 1) && v37 == *(_QWORD *)&v14[32 * (1 - v10)]
          || v37 == *(_QWORD *)&v14[-32 * v10] )
        {
          v38 = *(_QWORD *)&v14[32 * (2 - v10)];
          if ( *(_BYTE *)v38 != 17 )
            goto LABEL_27;
          v39 = *(_QWORD **)(v38 + 24);
          if ( *(_DWORD *)(v38 + 32) > 0x40u )
            v39 = (_QWORD *)*v39;
          v20 = sub_2F83220(v8, v37, (__int64)v39, v13, a3);
          if ( !(_BYTE)v20 )
            goto LABEL_72;
        }
        goto LABEL_10;
      }
      if ( *v14 == 89 )
        goto LABEL_10;
    }
    else
    {
      switch ( *v14 )
      {
        case 0x22u:
          if ( !sub_B46A10(v12[3]) )
          {
            v10 = *((_DWORD *)v14 + 1) & 0x7FFFFFF;
LABEL_41:
            v10 *= 32;
            v50 = &v14[-v10];
            v51 = -32;
            if ( v15 != 56 )
              v51 = -96;
            if ( (v14[7] & 0x80u) != 0 )
            {
              v28 = sub_BD2BC0((__int64)v14);
              v29 = v28 + v10;
              if ( (v14[7] & 0x80u) == 0 )
              {
                if ( !(unsigned int)(v29 >> 4) )
                  goto LABEL_52;
              }
              else
              {
                if ( !(unsigned int)((v29 - sub_BD2BC0((__int64)v14)) >> 4) )
                  goto LABEL_52;
                if ( (v14[7] & 0x80u) != 0 )
                {
                  v30 = *(_DWORD *)(sub_BD2BC0((__int64)v14) + 8);
                  if ( (v14[7] & 0x80u) == 0 )
                    BUG();
                  v31 = sub_BD2BC0((__int64)v14);
                  v51 -= 32LL * (unsigned int)(*(_DWORD *)(v31 + v10 - 4) - v30);
                  goto LABEL_52;
                }
              }
              BUG();
            }
LABEL_52:
            v52 = &v14[v51];
            if ( v50 == v52 )
              goto LABEL_10;
            v48 = v8;
            v32 = v14;
            v47 = v13;
            v33 = v50;
            while ( 2 )
            {
              if ( v53 != *(_QWORD *)v33 )
                goto LABEL_54;
              v34 = (v33 - v50) >> 5;
              if ( sub_B49EE0(v32, v34) )
                goto LABEL_27;
              v35 = *v32;
              if ( v35 == 40 )
              {
                v49 = 32LL * (unsigned int)sub_B491D0((__int64)v32);
              }
              else
              {
                v49 = 0;
                if ( v35 != 85 )
                {
                  if ( v35 != 34 )
                    BUG();
                  v49 = 64;
                }
              }
              if ( (v32[7] & 0x80u) != 0 )
              {
                v40 = sub_BD2BC0((__int64)v32);
                v42 = v40 + v41;
                if ( (v32[7] & 0x80u) == 0 )
                {
                  if ( (unsigned int)(v42 >> 4) )
LABEL_98:
                    BUG();
                }
                else if ( (unsigned int)((v42 - sub_BD2BC0((__int64)v32)) >> 4) )
                {
                  if ( (v32[7] & 0x80u) == 0 )
                    goto LABEL_98;
                  v43 = *(_DWORD *)(sub_BD2BC0((__int64)v32) + 8);
                  if ( (v32[7] & 0x80u) == 0 )
                    BUG();
                  v44 = sub_BD2BC0((__int64)v32);
                  v46 = 32LL * (unsigned int)(*(_DWORD *)(v44 + v45 - 4) - v43);
LABEL_83:
                  if ( (unsigned int)v34 >= (unsigned int)((32LL * (*((_DWORD *)v32 + 1) & 0x7FFFFFF) - 32 - v49 - v46) >> 5) )
                  {
                    sub_B49810((__int64)v32, v34);
LABEL_87:
                    LOBYTE(v20) = sub_B49E00((__int64)v32);
                    if ( !(_BYTE)v20 )
                      goto LABEL_72;
                  }
                  else if ( !(unsigned __int8)sub_B49B80((__int64)v32, v34, 50) )
                  {
                    goto LABEL_87;
                  }
LABEL_54:
                  v33 += 32;
                  if ( v33 == v52 )
                  {
                    v8 = v48;
                    v13 = v47;
                    goto LABEL_10;
                  }
                  continue;
                }
              }
              break;
            }
            v46 = 0;
            goto LABEL_83;
          }
          goto LABEL_10;
        case 0x3Du:
          v16 = *((_QWORD *)v14 + 1);
          goto LABEL_9;
        case 0x1Eu:
          goto LABEL_27;
      }
    }
    if ( !v64 )
      goto LABEL_44;
    v22 = v61;
    v11 = HIDWORD(v62);
    v10 = (__int64)&v61[HIDWORD(v62)];
    if ( v61 != (unsigned __int8 **)v10 )
    {
      while ( v14 != *v22 )
      {
        if ( (unsigned __int8 **)v10 == ++v22 )
          goto LABEL_22;
      }
      goto LABEL_10;
    }
LABEL_22:
    if ( HIDWORD(v62) < (unsigned int)v62 )
    {
      ++HIDWORD(v62);
      *(_QWORD *)v10 = v14;
      ++v60;
    }
    else
    {
LABEL_44:
      sub_C8CC70((__int64)&v60, v12[3], v10, v11, a5, a6);
      if ( !(_BYTE)v10 )
        goto LABEL_10;
    }
    v23 = (unsigned int)v58;
    v11 = HIDWORD(v58);
    v24 = (unsigned int)v58 + 1LL;
    if ( v24 > HIDWORD(v58) )
    {
      sub_C8D5F0((__int64)&v57, v59, v24, 8u, a5, a6);
      v23 = (unsigned int)v58;
    }
    v10 = (__int64)v57;
    v57[v23] = v14;
    LODWORD(v58) = v58 + 1;
LABEL_10:
    v12 = (__int64 *)v12[1];
    if ( !v12 )
    {
      v9 = v58;
      v7 = v57;
      goto LABEL_12;
    }
  }
  if ( (v14[7] & 0x40) != 0 )
    v26 = (unsigned __int8 *)*((_QWORD *)v14 - 1);
  else
    v26 = &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
  if ( v53 == *(_QWORD *)v26 )
  {
LABEL_27:
    v7 = v57;
    v21 = 0;
    goto LABEL_28;
  }
  v16 = *(_QWORD *)(*(_QWORD *)v26 + 8LL);
LABEL_9:
  v17 = sub_9208B0(*(_QWORD *)(v8 + 16), v16);
  v56[1] = v18;
  v56[0] = (unsigned __int64)(v17 + 7) >> 3;
  v19 = sub_CA1930(v56);
  v20 = sub_2F83220(v8, *v12, v19, v13, a3);
  if ( (_BYTE)v20 )
    goto LABEL_10;
LABEL_72:
  v7 = v57;
  v21 = v20;
LABEL_28:
  if ( v7 != v59 )
    _libc_free((unsigned __int64)v7);
  if ( !v64 )
    _libc_free((unsigned __int64)v61);
  return v21;
}
