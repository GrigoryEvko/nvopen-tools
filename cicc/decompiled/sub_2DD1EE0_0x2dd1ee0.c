// Function: sub_2DD1EE0
// Address: 0x2dd1ee0
//
__int64 __fastcall sub_2DD1EE0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 *v7; // r14
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // r12
  __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int8 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 i; // rbx
  char v25; // al
  __int64 v26; // rax
  char v27; // cl
  __int64 *v28; // r15
  __int64 *v29; // rbx
  int v30; // r13d
  __int64 v31; // r14
  __int16 *v32; // rax
  __int16 *v33; // rdx
  unsigned __int8 *v34; // rax
  __int16 *v35; // rdx
  __int16 *v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // r12
  __int64 v41; // r13
  _QWORD *v42; // rdi
  __int64 v43; // [rsp+28h] [rbp-1F8h]
  __int64 v44; // [rsp+30h] [rbp-1F0h]
  __int64 v45; // [rsp+30h] [rbp-1F0h]
  _QWORD *v46; // [rsp+30h] [rbp-1F0h]
  __int64 v47; // [rsp+38h] [rbp-1E8h]
  unsigned __int8 v48; // [rsp+38h] [rbp-1E8h]
  __int64 v49; // [rsp+40h] [rbp-1E0h] BYREF
  __int16 *v50; // [rsp+48h] [rbp-1D8h]
  __int64 v51; // [rsp+50h] [rbp-1D0h]
  int v52; // [rsp+58h] [rbp-1C8h]
  char v53; // [rsp+5Ch] [rbp-1C4h]
  __int16 v54; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 *v55; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v56; // [rsp+E8h] [rbp-138h]
  _BYTE v57[304]; // [rsp+F0h] [rbp-130h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 80);
  v55 = (__int64 *)v57;
  v56 = 0x2000000000LL;
  if ( v3 == a1 + 72 )
    return v2;
  v47 = v3;
  do
  {
    if ( !v47 )
      BUG();
    v4 = *(_QWORD *)(v47 + 32);
    while ( v47 + 24 != v4 )
    {
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 8);
      if ( *(_BYTE *)(v5 - 24) == 85 )
      {
        v9 = *(_QWORD *)(v5 - 56);
        if ( v9 )
        {
          if ( !*(_BYTE *)v9 )
          {
            a2 = *(_QWORD *)(v5 + 56);
            if ( *(_QWORD *)(v9 + 24) == a2 && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
            {
              v10 = *(_DWORD *)(v9 + 36);
              switch ( v10 )
              {
                case 183:
                  v20 = sub_BD3990(*(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) - 24), a2);
                  v21 = (unsigned int)v56;
                  v22 = (unsigned int)v56 + 1LL;
                  if ( v22 > HIDWORD(v56) )
                  {
                    a2 = (__int64)v57;
                    sub_C8D5F0((__int64)&v55, v57, v22, 8u, v18, v19);
                    v21 = (unsigned int)v56;
                  }
                  v55[v21] = (__int64)v20;
                  LODWORD(v56) = v56 + 1;
                  break;
                case 184:
                  v14 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) - 24);
                  v45 = *(_QWORD *)(v5 - 24 + 32 * (2LL - (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)));
                  v15 = sub_BD2C40(80, unk_3F10A10);
                  v16 = (__int64)v15;
                  if ( v15 )
                  {
                    v17 = v45;
                    v46 = v15;
                    sub_B4D460((__int64)v15, v14, v17, v5, 0);
                    v16 = (__int64)v46;
                  }
                  a2 = v16;
                  sub_BD84D0(v5 - 24, v16);
                  v2 = 1;
                  sub_B43D60((_QWORD *)(v5 - 24));
                  break;
                case 182:
                  v43 = *(_QWORD *)(v5 - 16);
                  v11 = *(_QWORD *)(v5 - 24 + 32 * (1LL - (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)));
                  v54 = 257;
                  v44 = v11;
                  v12 = (unsigned __int8 *)sub_BD2C40(80, 1u);
                  v13 = v12;
                  if ( v12 )
                    sub_B4D230((__int64)v12, v43, v44, (__int64)&v49, v5, 0);
                  sub_BD6B90(v13, (unsigned __int8 *)(v5 - 24));
                  a2 = (__int64)v13;
                  v2 = 1;
                  sub_BD84D0(v5 - 24, a2);
                  sub_B43D60((_QWORD *)(v5 - 24));
                  break;
              }
            }
          }
        }
      }
    }
    v47 = *(_QWORD *)(v47 + 8);
  }
  while ( a1 + 72 != v47 );
  v6 = (unsigned int)v56;
  v7 = v55;
  if ( (_DWORD)v56 )
  {
    v23 = *(_QWORD *)(a1 + 80);
    if ( !v23 )
      BUG();
    for ( i = *(_QWORD *)(v23 + 32); ; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 24) != 60 )
        break;
    }
    v49 = 0;
    v50 = &v54;
    v51 = 16;
    v52 = 0;
    v53 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v25 = *(_BYTE *)(i - 24);
        if ( v25 != 60 )
          break;
LABEL_52:
        i = *(_QWORD *)(i + 8);
        if ( !i )
          goto LABEL_76;
      }
      if ( (unsigned __int8)(v25 - 61) > 2u )
      {
        if ( v25 == 85 )
        {
          v26 = *(_QWORD *)(i - 56);
          if ( v26 )
          {
            if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(i + 56) && *(_DWORD *)(v26 + 36) == 183 )
              goto LABEL_52;
          }
        }
        v48 = v2;
        v27 = v53;
        v28 = &v7[v6];
        v29 = v7;
        v30 = 0;
        while ( 2 )
        {
          while ( 2 )
          {
            while ( 1 )
            {
              v31 = *v29;
              if ( !v27 )
                break;
              v32 = v50;
              v33 = &v50[4 * HIDWORD(v51)];
              if ( v50 == v33 )
                goto LABEL_60;
              while ( v31 != *(_QWORD *)v32 )
              {
                v32 += 4;
                if ( v33 == v32 )
                  goto LABEL_60;
              }
              if ( v28 == ++v29 )
              {
LABEL_47:
                if ( !v27 )
                  _libc_free((unsigned __int64)v50);
                v7 = v55;
                v2 = v30 | v48;
                goto LABEL_9;
              }
            }
            if ( sub_C8CA60((__int64)&v49, *v29) )
            {
              ++v29;
              v27 = v53;
              if ( v28 == v29 )
                goto LABEL_47;
              continue;
            }
            break;
          }
LABEL_60:
          v39 = sub_AC9EC0(*(__int64 ***)(v31 + 72));
          v40 = *(_QWORD *)(v31 + 32);
          v41 = v39;
          v42 = sub_BD2C40(80, unk_3F10A10);
          if ( v42 )
            sub_B4D460((__int64)v42, v41, v31, v40, 0);
          ++v29;
          v27 = v53;
          v30 = 1;
          if ( v28 == v29 )
            goto LABEL_47;
          continue;
        }
      }
      if ( v25 != 62 )
        goto LABEL_52;
      v34 = sub_BD3990(*(unsigned __int8 **)(i - 56), a2);
      if ( *v34 != 60 )
        goto LABEL_52;
      if ( v53 )
      {
        v35 = v50;
        a2 = HIDWORD(v51);
        v36 = &v50[4 * HIDWORD(v51)];
        if ( v50 != v36 )
        {
          while ( v34 != *(unsigned __int8 **)v35 )
          {
            v35 += 4;
            if ( v36 == v35 )
              goto LABEL_67;
          }
          goto LABEL_52;
        }
LABEL_67:
        if ( HIDWORD(v51) < (unsigned int)v51 )
        {
          a2 = (unsigned int)++HIDWORD(v51);
          *(_QWORD *)v36 = v34;
          ++v49;
          goto LABEL_52;
        }
      }
      a2 = (__int64)v34;
      sub_C8CC70((__int64)&v49, (__int64)v34, (__int64)v35, (__int64)v36, v37, v38);
      i = *(_QWORD *)(i + 8);
      if ( !i )
LABEL_76:
        BUG();
    }
  }
LABEL_9:
  if ( v7 != (__int64 *)v57 )
    _libc_free((unsigned __int64)v7);
  return v2;
}
