// Function: sub_1AF4F90
// Address: 0x1af4f90
//
void __fastcall sub_1AF4F90(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  char v4; // cl
  __int64 v5; // rbx
  __int64 v6; // rax
  void *v7; // rax
  __int64 v8; // rax
  __int64 *v9; // r13
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r12
  _QWORD *v16; // rdx
  _QWORD *v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned int v23; // esi
  _QWORD *v24; // rcx
  __int64 v25; // r10
  __int64 v26; // rax
  __int64 v27; // r13
  int v28; // ecx
  int v29; // edx
  _QWORD *v30; // rbx
  _QWORD *v31; // r13
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // r13
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rbx
  _QWORD *v42; // r12
  __int64 v43; // rsi
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  __int64 v46; // rsi
  __int64 *v47; // [rsp-108h] [rbp-108h]
  __int64 *v48; // [rsp-100h] [rbp-100h]
  __int64 v49; // [rsp-F0h] [rbp-F0h]
  void *v50; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v51; // [rsp-E0h] [rbp-E0h] BYREF
  __int64 v52; // [rsp-D8h] [rbp-D8h]
  __int64 v53; // [rsp-D0h] [rbp-D0h]
  __int64 v54; // [rsp-C8h] [rbp-C8h]
  void *v55; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v56; // [rsp-B0h] [rbp-B0h] BYREF
  __int64 v57; // [rsp-A8h] [rbp-A8h]
  __int64 v58; // [rsp-A0h] [rbp-A0h]
  __int64 i; // [rsp-98h] [rbp-98h]
  __int64 v60; // [rsp-88h] [rbp-88h] BYREF
  _QWORD *v61; // [rsp-80h] [rbp-80h]
  __int64 v62; // [rsp-78h] [rbp-78h]
  unsigned int v63; // [rsp-70h] [rbp-70h]
  _QWORD *v64; // [rsp-60h] [rbp-60h]
  unsigned int v65; // [rsp-50h] [rbp-50h]
  char v66; // [rsp-48h] [rbp-48h]
  char v67; // [rsp-3Fh] [rbp-3Fh]

  if ( *(_DWORD *)(a2 + 8) )
  {
    v60 = 0;
    v63 = 128;
    v2 = (_QWORD *)sub_22077B0(0x2000);
    v62 = 0;
    v61 = v2;
    v56 = 2;
    v57 = 0;
    v58 = -8;
    v3 = v2 + 1024;
    v55 = &unk_49E6B50;
    for ( i = 0; v3 != v2; v2 += 8 )
    {
      if ( v2 )
      {
        v4 = v56;
        v2[2] = 0;
        v2[3] = -8;
        *v2 = &unk_49E6B50;
        v2[1] = v4 & 6;
        v2[4] = i;
      }
    }
    v5 = *(_QWORD *)(a1 + 48);
    v66 = 0;
    v67 = 1;
    if ( v5 == a1 + 40 )
    {
      if ( (_DWORD)v62 )
        goto LABEL_23;
    }
    else
    {
      do
      {
        if ( !v5 )
          BUG();
        if ( *(_BYTE *)(v5 - 8) == 78 )
        {
          v6 = *(_QWORD *)(v5 - 48);
          if ( !*(_BYTE *)(v6 + 16)
            && (*(_BYTE *)(v6 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v6 + 36) - 35) <= 3 )
          {
            v7 = (void *)sub_1601A30(v5 - 24, 1);
            if ( v7 )
            {
              if ( *((_BYTE *)v7 + 16) == 77 )
              {
                v50 = v7;
                v51 = 6;
                v52 = 0;
                v53 = v5 - 24;
                if ( v5 != 8 && v5 != 16 )
                  sub_164C220((__int64)&v51);
                sub_1AF4A70((__int64)&v55, (__int64)&v60, (__int64 *)&v50);
                if ( v53 != -8 && v53 != 0 && v53 != -16 )
                  sub_1649B30(&v51);
              }
            }
          }
        }
        v5 = *(_QWORD *)(v5 + 8);
      }
      while ( a1 + 40 != v5 );
      if ( (_DWORD)v62 )
      {
LABEL_23:
        v8 = sub_157E9C0(a1);
        v9 = *(__int64 **)a2;
        v48 = (__int64 *)v8;
        v49 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v49 )
        {
          do
          {
            v10 = *v9;
            v11 = *(_QWORD *)(*v9 + 40);
            v12 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v11) + 16) - 34;
            if ( (unsigned int)v12 > 0x36 || (v13 = 0x40018000000001LL, !_bittest64(&v13, v12)) )
            {
              v14 = sub_1624210(v10);
              v15 = sub_1628DA0(v48, (__int64)v14);
              if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
              {
                v16 = *(_QWORD **)(v10 - 8);
                v10 = (__int64)&v16[3 * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)];
              }
              else
              {
                v16 = (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
              }
              v17 = v16;
              if ( v16 != (_QWORD *)v10 )
              {
                v47 = v9;
                do
                {
                  if ( v63 )
                  {
                    v23 = (v63 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
                    v24 = &v61[8 * (unsigned __int64)v23];
                    v25 = v24[3];
                    if ( *v17 == v25 )
                    {
LABEL_44:
                      if ( v24 != &v61[8 * (unsigned __int64)v63] )
                      {
                        v26 = sub_15F4880(v24[7]);
                        v27 = v26;
                        if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
                          v18 = *(__int64 **)(v26 - 8);
                        else
                          v18 = (__int64 *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
                        if ( *v18 )
                        {
                          v19 = v18[1];
                          v20 = v18[2] & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v20 = v19;
                          if ( v19 )
                            *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
                        }
                        *v18 = v15;
                        if ( v15 )
                        {
                          v21 = *(_QWORD *)(v15 + 8);
                          v18[1] = v21;
                          if ( v21 )
                            *(_QWORD *)(v21 + 16) = (unsigned __int64)(v18 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
                          v18[2] = (v15 + 8) | v18[2] & 3;
                          *(_QWORD *)(v15 + 8) = v18;
                        }
                        v22 = sub_157EE30(v11);
                        if ( v22 )
                          v22 -= 24;
                        sub_15F2120(v27, v22);
                      }
                    }
                    else
                    {
                      v28 = 1;
                      while ( v25 != -8 )
                      {
                        v29 = v28 + 1;
                        v23 = (v63 - 1) & (v28 + v23);
                        v24 = &v61[8 * (unsigned __int64)v23];
                        v25 = v24[3];
                        if ( *v17 == v25 )
                          goto LABEL_44;
                        v28 = v29;
                      }
                    }
                  }
                  v17 += 3;
                }
                while ( (_QWORD *)v10 != v17 );
                v9 = v47;
              }
            }
            ++v9;
          }
          while ( (__int64 *)v49 != v9 );
        }
        if ( v66 )
        {
          if ( v65 )
          {
            v41 = v64;
            v42 = &v64[2 * v65];
            do
            {
              if ( *v41 != -8 && *v41 != -4 )
              {
                v43 = v41[1];
                if ( v43 )
                  sub_161E7C0((__int64)(v41 + 1), v43);
              }
              v41 += 2;
            }
            while ( v42 != v41 );
          }
          j___libc_free_0(v64);
        }
        if ( !v63 )
          goto LABEL_51;
        v30 = v61;
        v51 = 2;
        v52 = 0;
        v31 = &v61[8 * (unsigned __int64)v63];
        v53 = -8;
        v32 = -8;
        v50 = &unk_49E6B50;
        v54 = 0;
        v56 = 2;
        v57 = 0;
        v58 = -16;
        v55 = &unk_49E6B50;
        i = 0;
        while ( 1 )
        {
          v33 = v30[3];
          if ( v32 != v33 )
          {
            v32 = v58;
            if ( v33 != v58 )
            {
              v34 = v30[7];
              if ( v34 != -8 && v34 != 0 && v34 != -16 )
              {
                sub_1649B30(v30 + 5);
                v33 = v30[3];
              }
              v32 = v33;
            }
          }
          *v30 = &unk_49EE2B0;
          if ( v32 != -8 && v32 != 0 && v32 != -16 )
            sub_1649B30(v30 + 1);
          v30 += 8;
          if ( v31 == v30 )
            break;
          v32 = v53;
        }
        v55 = &unk_49EE2B0;
        if ( v58 != -8 && v58 != 0 && v58 != -16 )
          sub_1649B30(&v56);
        v50 = &unk_49EE2B0;
        v35 = v53;
        if ( v53 == -8 || v53 == 0 )
          goto LABEL_51;
LABEL_73:
        if ( v35 != -16 )
          sub_1649B30(&v51);
LABEL_51:
        j___libc_free_0(v61);
        return;
      }
      if ( v66 )
      {
        if ( v65 )
        {
          v44 = v64;
          v45 = &v64[2 * v65];
          do
          {
            if ( *v44 != -8 && *v44 != -4 )
            {
              v46 = v44[1];
              if ( v46 )
                sub_161E7C0((__int64)(v44 + 1), v46);
            }
            v44 += 2;
          }
          while ( v45 != v44 );
        }
        j___libc_free_0(v64);
      }
    }
    if ( !v63 )
      goto LABEL_51;
    v36 = v61;
    v51 = 2;
    v52 = 0;
    v37 = &v61[8 * (unsigned __int64)v63];
    v53 = -8;
    v38 = -8;
    v50 = &unk_49E6B50;
    v54 = 0;
    v56 = 2;
    v57 = 0;
    v58 = -16;
    v55 = &unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v39 = v36[3];
      if ( v39 != v38 )
      {
        v38 = v58;
        if ( v39 != v58 )
        {
          v40 = v36[7];
          if ( v40 != -8 && v40 != 0 && v40 != -16 )
          {
            sub_1649B30(v36 + 5);
            v39 = v36[3];
          }
          v38 = v39;
        }
      }
      *v36 = &unk_49EE2B0;
      if ( v38 != 0 && v38 != -8 && v38 != -16 )
        sub_1649B30(v36 + 1);
      v36 += 8;
      if ( v37 == v36 )
        break;
      v38 = v53;
    }
    v55 = &unk_49EE2B0;
    if ( v58 != 0 && v58 != -8 && v58 != -16 )
      sub_1649B30(&v56);
    v50 = &unk_49EE2B0;
    v35 = v53;
    if ( v53 == 0 || v53 == -8 )
      goto LABEL_51;
    goto LABEL_73;
  }
}
