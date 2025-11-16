// Function: sub_1654C20
// Address: 0x1654c20
//
void __fastcall sub_1654C20(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  char v6; // al
  __int64 v7; // r14
  _QWORD *v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  _QWORD *v20; // rax
  unsigned __int64 v21; // r14
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  unsigned int *v29; // rax
  __int64 v30; // r8
  __int64 *v31; // rax
  unsigned __int64 v32; // rdi
  char v33; // dl
  char v34; // r9
  __int64 *v35; // rdx
  __int64 *v36; // r8
  __int64 *v37; // rsi
  __int64 *v38; // rcx
  __int64 *v39; // rax
  __int64 v40; // r12
  _BYTE *v41; // rax
  bool v42; // zf
  __int64 v43; // [rsp+0h] [rbp-E0h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  __int64 v45; // [rsp+18h] [rbp-C8h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  __int64 v48[2]; // [rsp+20h] [rbp-C0h] BYREF
  char v49; // [rsp+30h] [rbp-B0h]
  char v50; // [rsp+31h] [rbp-AFh]
  const char *v51; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v52; // [rsp+48h] [rbp-98h]
  __int64 *v53; // [rsp+50h] [rbp-90h]
  __int64 v54; // [rsp+58h] [rbp-88h]
  int v55; // [rsp+60h] [rbp-80h]
  _BYTE v56[120]; // [rsp+68h] [rbp-78h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 80LL);
  if ( v5 && v4 == v5 - 24 )
  {
    v48[0] = a2;
    v51 = "EH pad cannot be in entry block.";
    LOWORD(v53) = 259;
    sub_1654980(a1, (__int64)&v51, v48);
    return;
  }
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 != 88 )
  {
    if ( v6 == 74 )
    {
      v13 = *(_QWORD *)(v4 + 8);
      if ( v13 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v13) + 16) - 25) > 9u )
        {
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            goto LABEL_32;
        }
        v14 = sub_157F120(v4);
        v15 = *(_QWORD *)(a2 - 24);
        if ( *(_QWORD *)(v15 + 40) != v14 )
        {
          v51 = "Block containg CatchPadInst must be jumped to only by its catchswitch.";
          LOWORD(v53) = 259;
          sub_164FF40(a1, (__int64)&v51);
          if ( !*a1 )
            return;
          goto LABEL_16;
        }
      }
      else
      {
LABEL_32:
        v15 = *(_QWORD *)(a2 - 24);
      }
      if ( (*(_BYTE *)(v15 + 18) & 1) == 0 )
        return;
      v16 = (*(_BYTE *)(v15 + 23) & 0x40) != 0 ? *(_QWORD *)(v15 - 8) : v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
      v17 = *(_QWORD *)(v16 + 24);
      if ( v4 != v17 )
        return;
      if ( !v17 )
        return;
      v51 = "Catchswitch cannot unwind to one of its catchpads";
      LOWORD(v53) = 259;
      sub_164FF40(a1, (__int64)&v51);
      if ( !*a1 )
        return;
      sub_164FA80(a1, v15);
LABEL_16:
      sub_164FA80(a1, a2);
      return;
    }
    v18 = sub_164ED90(a2);
    v19 = *(_QWORD *)(v4 + 8);
    v44 = v18;
    if ( !v19 )
      return;
    while ( 1 )
    {
      v20 = sub_1648700(v19);
      if ( (unsigned __int8)(*((_BYTE *)v20 + 16) - 25) <= 9u )
        break;
      v19 = *(_QWORD *)(v19 + 8);
      if ( !v19 )
        return;
    }
LABEL_37:
    v21 = sub_157EBA0(v20[5]);
    v22 = *(_BYTE *)(v21 + 16);
    switch ( v22 )
    {
      case 29:
        if ( v4 == *(_QWORD *)(v21 - 24) && v4 != *(_QWORD *)(v21 - 48) )
        {
          if ( *(char *)(v21 + 23) >= 0 )
            goto LABEL_89;
          v23 = sub_1648A40(v21);
          v25 = v24 + v23;
          if ( *(char *)(v21 + 23) < 0 )
            v25 -= sub_1648A40(v21);
          v26 = v25 >> 4;
          if ( (_DWORD)v26 )
          {
            v27 = 0;
            v43 = 16LL * (unsigned int)v26;
            while ( 1 )
            {
              v28 = 0;
              if ( *(char *)(v21 + 23) < 0 )
              {
                v45 = v27;
                v28 = sub_1648A40(v21);
                v27 = v45;
              }
              v29 = (unsigned int *)(v27 + v28);
              if ( *(_DWORD *)(*(_QWORD *)v29 + 8LL) == 1 )
                break;
              v27 += 16;
              if ( v43 == v27 )
                goto LABEL_89;
            }
            v30 = *(_QWORD *)(v21 + 24 * (v29[2] - (unsigned __int64)(*(_DWORD *)(v21 + 20) & 0xFFFFFFF)));
          }
          else
          {
LABEL_89:
            v39 = (__int64 *)sub_16498A0(v21);
            v30 = sub_1594470(v39);
          }
LABEL_50:
          v55 = 0;
          v51 = 0;
          v52 = (__int64 *)v56;
          v53 = (__int64 *)v56;
          v54 = 8;
          if ( v3 != v30 )
          {
            if ( v44 == v30 )
              goto LABEL_65;
            do
            {
LABEL_56:
              if ( *(_BYTE *)(v30 + 16) == 16 )
              {
                v50 = 1;
                v48[0] = (__int64)"A single unwind edge may only enter one EH pad";
                v49 = 3;
                sub_164FF40(a1, (__int64)v48);
                if ( *a1 )
                  sub_164FA80(a1, v21);
                goto LABEL_76;
              }
              v31 = v52;
              if ( v53 != v52 )
                goto LABEL_58;
              v37 = &v52[HIDWORD(v54)];
              if ( v52 != v37 )
              {
                v38 = 0;
                while ( v30 != *v31 )
                {
                  if ( *v31 == -2 )
                    v38 = v31;
                  if ( v37 == ++v31 )
                  {
                    if ( !v38 )
                      goto LABEL_84;
                    *v38 = v30;
                    v32 = (unsigned __int64)v53;
                    --v55;
                    v35 = v52;
                    ++v51;
                    goto LABEL_59;
                  }
                }
LABEL_74:
                v47 = v30;
                v50 = 1;
                v48[0] = (__int64)"EH pad jumps through a cycle of pads";
                v49 = 3;
                sub_164FF40(a1, (__int64)v48);
                if ( *a1 )
                  sub_164FA80(a1, v47);
                goto LABEL_76;
              }
LABEL_84:
              if ( HIDWORD(v54) < (unsigned int)v54 )
              {
                ++HIDWORD(v54);
                *v37 = v30;
                v35 = v52;
                ++v51;
                v32 = (unsigned __int64)v53;
              }
              else
              {
LABEL_58:
                v46 = v30;
                sub_16CCBA0(&v51, v30);
                v32 = (unsigned __int64)v53;
                v30 = v46;
                v34 = v33;
                v35 = v52;
                if ( !v34 )
                  goto LABEL_74;
              }
LABEL_59:
              if ( (unsigned __int8)(*(_BYTE *)(v30 + 16) - 73) <= 1u )
              {
                v30 = *(_QWORD *)(v30 - 24);
              }
              else
              {
                if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
                  v36 = *(__int64 **)(v30 - 8);
                else
                  v36 = (__int64 *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
                v30 = *v36;
              }
              if ( v30 == v3 )
                goto LABEL_94;
            }
            while ( v30 != v44 );
            if ( (__int64 *)v32 != v35 )
              _libc_free(v32);
LABEL_65:
            while ( 1 )
            {
              v19 = *(_QWORD *)(v19 + 8);
              if ( !v19 )
                return;
              v20 = sub_1648700(v19);
              if ( (unsigned __int8)(*((_BYTE *)v20 + 16) - 25) <= 9u )
                goto LABEL_37;
            }
          }
LABEL_93:
          v3 = v30;
LABEL_94:
          v50 = 1;
          v48[0] = (__int64)"EH pad cannot handle exceptions raised within it";
          v49 = 3;
          sub_164FF40(a1, (__int64)v48);
          if ( *a1 )
          {
            sub_164FA80(a1, v3);
            sub_164FA80(a1, v21);
          }
LABEL_76:
          if ( v53 != v52 )
            _libc_free((unsigned __int64)v53);
          return;
        }
        break;
      case 32:
        v30 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
        if ( v44 != v30 )
        {
          v55 = 0;
          v51 = 0;
          v52 = (__int64 *)v56;
          v53 = (__int64 *)v56;
          v54 = 8;
          if ( v3 != v30 )
            goto LABEL_56;
          goto LABEL_93;
        }
        v40 = *a1;
        v51 = "A cleanupret must exit its cleanup";
        LOWORD(v53) = 259;
        if ( v40 )
        {
          sub_16E2CE0(&v51, v40);
          v41 = *(_BYTE **)(v40 + 24);
          if ( (unsigned __int64)v41 >= *(_QWORD *)(v40 + 16) )
          {
            sub_16E7DE0(v40, 10);
          }
          else
          {
            *(_QWORD *)(v40 + 24) = v41 + 1;
            *v41 = 10;
          }
        }
        v42 = *a1 == 0;
        *((_BYTE *)a1 + 72) = 1;
        if ( v42 )
          return;
LABEL_102:
        sub_164FA80(a1, v21);
        return;
      case 34:
        v30 = v21;
        goto LABEL_50;
    }
    v51 = "EH pad must be jumped to via an unwind edge";
    LOWORD(v53) = 259;
    sub_164FF40(a1, (__int64)&v51);
    if ( !*a1 )
      return;
    sub_164FA80(a1, v3);
    goto LABEL_102;
  }
  v7 = *(_QWORD *)(v4 + 8);
  if ( !v7 )
    return;
  while ( 1 )
  {
    v8 = sub_1648700(v7);
    if ( (unsigned __int8)(*((_BYTE *)v8 + 16) - 25) <= 9u )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return;
  }
LABEL_11:
  v9 = sub_157EBA0(v8[5]);
  if ( *(_BYTE *)(v9 + 16) == 29 && v4 == *(_QWORD *)(v9 - 24) && v4 != *(_QWORD *)(v9 - 48) )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return;
      v8 = sub_1648700(v7);
      if ( (unsigned __int8)(*((_BYTE *)v8 + 16) - 25) <= 9u )
        goto LABEL_11;
    }
  }
  v10 = *a1;
  v51 = "Block containing LandingPadInst must be jumped to only by the unwind edge of an invoke.";
  LOWORD(v53) = 259;
  if ( !v10 )
  {
    *((_BYTE *)a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(&v51, v10);
  v11 = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
  {
    sub_16E7DE0(v10, 10);
  }
  else
  {
    *(_QWORD *)(v10 + 24) = v11 + 1;
    *v11 = 10;
  }
  v12 = *a1;
  *((_BYTE *)a1 + 72) = 1;
  if ( v12 )
    goto LABEL_16;
}
