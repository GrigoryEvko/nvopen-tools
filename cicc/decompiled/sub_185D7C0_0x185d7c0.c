// Function: sub_185D7C0
// Address: 0x185d7c0
//
__int64 __fastcall sub_185D7C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  unsigned int v4; // r13d
  __int64 v5; // r8
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rax
  int v14; // r9d
  unsigned __int8 v15; // al
  __int64 v16; // rax
  bool v18; // cl
  __int64 v19; // rax
  int v20; // eax
  bool v21; // zf
  unsigned __int64 v22; // rax
  char v23; // si
  unsigned __int64 v24; // r12
  __int64 *v25; // r12
  __int64 v26; // rsi
  unsigned __int64 v27; // rdi
  __int64 v28; // rsi
  unsigned int v29; // eax
  unsigned int v30; // edx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned __int64 v33; // rax
  __int64 *v34; // rsi
  __int64 v35; // rax
  _BYTE *v36; // r11
  __int64 v37; // rax
  unsigned int v38; // edx
  unsigned __int64 v39; // r8
  __int64 v40; // rcx
  __int64 v41; // r9
  __int64 v42; // r14
  unsigned int v43; // r15d
  __int64 **v44; // rbx
  __int64 v45; // r13
  __int64 v46; // r8
  __int64 *v47; // r12
  __int64 v48; // rcx
  unsigned __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // [rsp+8h] [rbp-B8h]
  __int64 v56; // [rsp+10h] [rbp-B0h]
  bool v57; // [rsp+18h] [rbp-A8h]
  _BYTE *v58; // [rsp+18h] [rbp-A8h]
  _BYTE *v59; // [rsp+18h] [rbp-A8h]
  _BYTE *v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+28h] [rbp-98h]
  _BYTE v62[8]; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v63; // [rsp+40h] [rbp-80h] BYREF
  __int64 v64; // [rsp+48h] [rbp-78h]
  _BYTE v65[112]; // [rsp+50h] [rbp-70h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = 0;
  v5 = *(_QWORD *)(a1 + 8);
  v61 = a2 + 8;
  if ( v5 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v5 + 8);
        v12 = (__int64)sub_1648700(v5);
        v13 = sub_15F2060(v12);
        if ( sub_15E4690(v13, 0) )
          return 0;
        v15 = *(_BYTE *)(v12 + 16);
        if ( v15 == 54 )
        {
          if ( *(_QWORD *)(v12 - 24) )
          {
            v6 = *(_QWORD *)(v12 - 16);
            v7 = *(_QWORD *)(v12 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v7 = v6;
            if ( v6 )
              *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
          }
          *(_QWORD *)(v12 - 24) = v2;
          if ( v2 )
          {
            v8 = *(_QWORD *)(v2 + 8);
            *(_QWORD *)(v12 - 16) = v8;
            if ( v8 )
              *(_QWORD *)(v8 + 16) = (v12 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
            v9 = *(_QWORD *)(v12 - 8);
            v10 = v12 - 24;
            *(_QWORD *)(v10 + 16) = v61 | v9 & 3;
            *(_QWORD *)(v2 + 8) = v10;
          }
          v5 = v11;
          v4 = 1;
          goto LABEL_11;
        }
        if ( v15 != 55 )
          break;
        v16 = *(_QWORD *)(v12 - 24);
        LOBYTE(v16) = v3 == v16 && v16 != 0;
        if ( !(_BYTE)v16 )
          goto LABEL_16;
        v48 = *(_QWORD *)(v12 - 16);
        v49 = *(_QWORD *)(v12 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v49 = v48;
        if ( v48 )
          *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
        *(_QWORD *)(v12 - 24) = v2;
        if ( v2 )
        {
          v50 = *(_QWORD *)(v2 + 8);
          *(_QWORD *)(v12 - 16) = v50;
          if ( v50 )
            *(_QWORD *)(v50 + 16) = (v12 - 16) | *(_QWORD *)(v50 + 16) & 3LL;
          v5 = v11;
          v4 = v16;
          *(_QWORD *)(v12 - 8) = v61 | *(_QWORD *)(v12 - 8) & 3LL;
          *(_QWORD *)(v2 + 8) = v12 - 24;
        }
        else
        {
          v5 = v11;
          v4 = v16;
        }
LABEL_11:
        if ( !v11 )
          return v4;
      }
      v18 = v15 == 29 || v15 == 78;
      if ( !v18 )
      {
        if ( (unsigned int)v15 - 60 > 0xC )
        {
          v5 = v11;
          if ( v15 == 56 )
          {
            v36 = v65;
            v64 = 0x800000000LL;
            v37 = 0;
            v63 = (unsigned __int64)v65;
            v38 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            v39 = v38 - 1;
            if ( v39 > 8 )
            {
              sub_16CD150((__int64)&v63, v65, v38 - 1, 8, v39, v14);
              v37 = (unsigned int)v64;
              v36 = v65;
              v38 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            }
            if ( v12 != v12 + 24 * (1LL - v38) )
            {
              v40 = v3;
              v41 = v2;
              v42 = v11;
              v43 = v4;
              v44 = (__int64 **)(v12 + 24 * (1LL - v38));
              v45 = v12;
              v46 = v40;
              do
              {
                v47 = *v44;
                if ( *((_BYTE *)*v44 + 16) > 0x10u )
                  break;
                if ( HIDWORD(v64) <= (unsigned int)v37 )
                {
                  v55 = v41;
                  v56 = v46;
                  v58 = v36;
                  sub_16CD150((__int64)&v63, v36, 0, 8, v46, v41);
                  v37 = (unsigned int)v64;
                  v41 = v55;
                  v46 = v56;
                  v36 = v58;
                }
                v44 += 3;
                *(_QWORD *)(v63 + 8 * v37) = v47;
                v37 = (unsigned int)(v64 + 1);
                LODWORD(v64) = v64 + 1;
              }
              while ( (__int64 **)v45 != v44 );
              v12 = v45;
              v11 = v42;
              v4 = v43;
              v3 = v46;
              v2 = v41;
              v38 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            }
            if ( v38 - 1 == (_DWORD)v37 )
            {
              v59 = v36;
              v62[4] = 0;
              v53 = sub_15A2E80(0, v2, (__int64 **)v63, (unsigned int)v37, 0, (__int64)v62, 0);
              v54 = sub_185D7C0(v12, v53);
              v36 = v59;
              v4 |= v54;
            }
            if ( !*(_QWORD *)(v12 + 8) )
            {
              v60 = v36;
              v4 = 1;
              sub_15F20C0((_QWORD *)v12);
              v36 = v60;
            }
            if ( (_BYTE *)v63 != v36 )
              _libc_free(v63);
            goto LABEL_16;
          }
        }
        else
        {
          v19 = sub_15A46C0((unsigned int)v15 - 24, (__int64 ***)v2, *(__int64 ***)v12, 0);
          v20 = sub_185D7C0(v12, v19);
          v5 = v11;
          v4 |= v20;
          if ( !*(_QWORD *)(v12 + 8) )
          {
            v4 = 1;
            sub_15F20C0((_QWORD *)v12);
            v5 = v11;
          }
        }
        goto LABEL_11;
      }
      if ( v15 <= 0x17u )
      {
        v63 = 0;
        v24 = 0;
      }
      else
      {
        v21 = v15 == 78;
        v22 = v12 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v21 )
        {
          v22 |= 4u;
          v23 = 1;
        }
        else
        {
          v23 = 0;
        }
        v24 = v12 & 0xFFFFFFFFFFFFFFF8LL;
        v63 = v22;
        if ( v23 )
        {
          if ( v3 == *(_QWORD *)(v24 - 24) )
          {
            v25 = (__int64 *)(v24 - 24);
LABEL_29:
            v26 = v25[1];
            v27 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v27 = v26;
            if ( v26 )
              *(_QWORD *)(v26 + 16) = v27 | *(_QWORD *)(v26 + 16) & 3LL;
            *v25 = v2;
            if ( v2 )
            {
              v28 = *(_QWORD *)(v2 + 8);
              v25[1] = v28;
              if ( v28 )
                *(_QWORD *)(v28 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v28 + 16) & 3LL;
              v25[2] = v61 | v25[2] & 3;
              *(_QWORD *)(v2 + 8) = v25;
            }
            v57 = v18;
            v29 = sub_165AFC0(&v63);
            if ( v29 )
            {
              v30 = 0;
              v31 = 0;
              v32 = 24LL * v29;
              do
              {
                v33 = v63 & 0xFFFFFFFFFFFFFFF8LL;
                v34 = (__int64 *)((v63 & 0xFFFFFFFFFFFFFFF8LL)
                                + v31
                                - 24LL * (*(_DWORD *)((v63 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
                if ( v3 == *v34 )
                {
                  if ( (*(_BYTE *)(v33 + 23) & 0x40) == 0 || (v34 = (__int64 *)(v31 + *(_QWORD *)(v33 - 8)), *v34) )
                  {
                    v51 = v34[1];
                    v52 = v34[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v52 = v51;
                    if ( v51 )
                      *(_QWORD *)(v51 + 16) = *(_QWORD *)(v51 + 16) & 3LL | v52;
                  }
                  *v34 = v2;
                  v30 = v57;
                  if ( v2 )
                  {
                    v35 = *(_QWORD *)(v2 + 8);
                    v34[1] = v35;
                    if ( v35 )
                      *(_QWORD *)(v35 + 16) = (unsigned __int64)(v34 + 1) | *(_QWORD *)(v35 + 16) & 3LL;
                    v30 = v57;
                    v34[2] = v61 | v34[2] & 3;
                    *(_QWORD *)(v2 + 8) = v34;
                  }
                }
                v31 += 24;
              }
              while ( v32 != v31 );
              v4 = v57;
              if ( (_BYTE)v30 )
              {
                v11 = *(_QWORD *)(v3 + 8);
                v4 = v30;
              }
            }
            else
            {
              v4 = v57;
            }
            goto LABEL_16;
          }
          goto LABEL_16;
        }
      }
      if ( v3 == *(_QWORD *)(v24 - 72) )
      {
        v25 = (__int64 *)(v24 - 72);
        goto LABEL_29;
      }
LABEL_16:
      v5 = v11;
      if ( !v11 )
        return v4;
    }
  }
  return 0;
}
