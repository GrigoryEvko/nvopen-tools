// Function: sub_1CA42C0
// Address: 0x1ca42c0
//
void __fastcall sub_1CA42C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // eax
  _QWORD *v7; // r12
  int v8; // r13d
  unsigned __int64 v9; // r15
  _QWORD *v10; // r14
  __int64 v11; // rax
  int v12; // r8d
  int v13; // r9d
  _QWORD *v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  unsigned __int8 v18; // dl
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int64 v21; // r14
  _QWORD *v22; // r15
  __int64 v23; // rax
  int v24; // esi
  int v25; // r11d
  __int64 *v26; // r10
  unsigned int v27; // edx
  __int64 *v28; // rdi
  __int64 v29; // rcx
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // r15
  unsigned int v33; // edx
  __int64 v34; // r14
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // r8
  unsigned __int64 v42; // rcx
  _QWORD *v43; // r15
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // r9d
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // [rsp+0h] [rbp-C0h]
  _QWORD *v53; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v54; // [rsp+18h] [rbp-A8h]
  _QWORD *v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+18h] [rbp-A8h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  __int64 v59; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v60; // [rsp+28h] [rbp-98h] BYREF
  __int64 v61; // [rsp+30h] [rbp-90h] BYREF
  __int64 v62; // [rsp+38h] [rbp-88h]
  __int64 v63; // [rsp+40h] [rbp-80h]
  __int64 v64; // [rsp+48h] [rbp-78h]
  _QWORD *v65; // [rsp+50h] [rbp-70h] BYREF
  __int64 v66; // [rsp+58h] [rbp-68h]
  _QWORD v67[12]; // [rsp+60h] [rbp-60h] BYREF

  v4 = (unsigned __int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v5 + 16) )
    BUG();
  v6 = *(_DWORD *)(v5 + 36);
  v7 = a1;
  if ( v6 == 4051 )
  {
    v8 = 5;
    goto LABEL_7;
  }
  if ( v6 <= 0xFD3 )
  {
    if ( v6 == 4048 )
    {
      v8 = 4;
    }
    else
    {
      if ( v6 != 4049 )
        return;
      v8 = 1;
    }
LABEL_7:
    v9 = *v4;
    if ( *(_BYTE *)(*(_QWORD *)*v4 + 8LL) != 15 )
      return;
    goto LABEL_11;
  }
  v8 = 3;
  if ( v6 == 4052 )
  {
    v9 = *v4;
    if ( *(_BYTE *)(*(_QWORD *)*v4 + 8LL) == 15 )
    {
LABEL_11:
      v10 = (_QWORD *)(*a1 + 120LL);
      v11 = sub_22077B0(32);
      if ( v11 )
        *(_QWORD *)v11 = 0;
      *(_QWORD *)(v11 + 8) = v9;
      *(_QWORD *)(v11 + 16) = a3;
      *(_DWORD *)(v11 + 24) = v8;
      sub_1C9EE80(v10, 0, (_QWORD *)(v11 + 8), v9, v11);
      v14 = v67;
      v66 = 0x600000001LL;
      LODWORD(v15) = 1;
      v65 = v67;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v67[0] = v9;
      while ( 1 )
      {
        v16 = (unsigned int)v15;
        v15 = (unsigned int)(v15 - 1);
        v17 = v14[v16 - 1];
        LODWORD(v66) = v15;
        v18 = *(_BYTE *)(v17 + 16);
        if ( v18 <= 0x17u )
          goto LABEL_18;
        if ( v18 == 71 )
        {
          v19 = *(_QWORD *)(v17 - 24);
          if ( (unsigned int)v15 >= HIDWORD(v66) )
          {
            sub_16CD150((__int64)&v65, v67, 0, 8, v12, v13);
            v14 = v65;
            v15 = (unsigned int)v66;
          }
          v14[v15] = v19;
          v20 = *v7;
          LODWORD(v66) = v66 + 1;
          v21 = *(_QWORD *)(v17 - 24);
          v22 = (_QWORD *)(v20 + 120);
          v23 = sub_22077B0(32);
          if ( v23 )
            *(_QWORD *)v23 = 0;
          *(_QWORD *)(v23 + 8) = v21;
          *(_QWORD *)(v23 + 16) = a3;
          *(_DWORD *)(v23 + 24) = v8;
          sub_1C9EE80(v22, 0, (_QWORD *)(v23 + 8), v21, v23);
          LODWORD(v15) = v66;
          if ( !(_DWORD)v66 )
          {
LABEL_27:
            j___libc_free_0(v62);
            if ( v65 != v67 )
              _libc_free((unsigned __int64)v65);
            return;
          }
          goto LABEL_19;
        }
        if ( v18 == 56 )
          break;
        if ( v18 == 79 )
        {
          v45 = *(_QWORD *)(v17 - 72);
          if ( (unsigned int)v15 >= HIDWORD(v66) )
          {
            v57 = *(_QWORD *)(v17 - 72);
            sub_16CD150((__int64)&v65, v67, 0, 8, v45, v13);
            v14 = v65;
            v15 = (unsigned int)v66;
            v45 = v57;
          }
          v14[v15] = v45;
          v46 = *v7;
          LODWORD(v66) = v66 + 1;
          v52 = *(_QWORD *)(v17 - 72);
          v55 = (_QWORD *)(v46 + 120);
          v47 = sub_22077B0(32);
          if ( v47 )
            *(_QWORD *)v47 = 0;
          *(_QWORD *)(v47 + 8) = v52;
          *(_QWORD *)(v47 + 16) = a3;
          *(_DWORD *)(v47 + 24) = v8;
          sub_1C9EE80(v55, 0, (_QWORD *)(v47 + 8), v52, v47);
          v49 = *(_QWORD *)(v17 - 48);
          v50 = (unsigned int)v66;
          if ( (unsigned int)v66 >= HIDWORD(v66) )
          {
            v58 = *(_QWORD *)(v17 - 48);
            sub_16CD150((__int64)&v65, v67, 0, 8, v49, v48);
            v50 = (unsigned int)v66;
            v49 = v58;
          }
          v65[v50] = v49;
          LODWORD(v66) = v66 + 1;
          v42 = *(_QWORD *)(v17 - 48);
LABEL_54:
          v54 = v42;
          v43 = (_QWORD *)(*v7 + 120LL);
          v44 = sub_22077B0(32);
          if ( v44 )
            *(_QWORD *)v44 = 0;
          *(_QWORD *)(v44 + 8) = v54;
          *(_QWORD *)(v44 + 16) = a3;
          *(_DWORD *)(v44 + 24) = v8;
          sub_1C9EE80(v43, 0, (_QWORD *)(v44 + 8), v54, v44);
          goto LABEL_57;
        }
        if ( v18 == 77 )
        {
          v24 = v64;
          v59 = v17;
          if ( (_DWORD)v64 )
          {
            v13 = v64 - 1;
            v25 = 1;
            v12 = v62;
            v26 = 0;
            v27 = (v64 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v28 = (__int64 *)(v62 + 8LL * v27);
            v29 = *v28;
            if ( v17 == *v28 )
              goto LABEL_18;
            while ( v29 != -8 )
            {
              if ( v26 || v29 != -16 )
                v28 = v26;
              v27 = v13 & (v25 + v27);
              v29 = *(_QWORD *)(v62 + 8LL * v27);
              if ( v17 == v29 )
                goto LABEL_18;
              ++v25;
              v26 = v28;
              v28 = (__int64 *)(v62 + 8LL * v27);
            }
            if ( !v26 )
              v26 = v28;
            ++v61;
            v30 = v63 + 1;
            if ( 4 * ((int)v63 + 1) < (unsigned int)(3 * v64) )
            {
              if ( (int)v64 - HIDWORD(v63) - v30 > (unsigned int)v64 >> 3 )
                goto LABEL_37;
              goto LABEL_74;
            }
          }
          else
          {
            ++v61;
          }
          v24 = 2 * v64;
LABEL_74:
          sub_176FD40((__int64)&v61, v24);
          sub_1A27740((__int64)&v61, &v59, &v60);
          v26 = v60;
          v17 = v59;
          v30 = v63 + 1;
LABEL_37:
          LODWORD(v63) = v30;
          if ( *v26 != -8 )
            --HIDWORD(v63);
          *v26 = v17;
          v31 = v59;
          v32 = 0;
          v33 = *(_DWORD *)(v59 + 20) & 0xFFFFFFF;
          if ( v33 )
          {
            v53 = v7;
            while ( 1 )
            {
              if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
              {
                v34 = 3 * v32;
                v35 = *(_QWORD *)(*(_QWORD *)(v31 - 8) + 24 * v32);
                v36 = (unsigned int)v66;
                if ( (unsigned int)v66 >= HIDWORD(v66) )
                  goto LABEL_49;
              }
              else
              {
                v34 = 3 * v32;
                v35 = *(_QWORD *)(v31 - 24LL * v33 + 24 * v32);
                v36 = (unsigned int)v66;
                if ( (unsigned int)v66 >= HIDWORD(v66) )
                {
LABEL_49:
                  sub_16CD150((__int64)&v65, v67, 0, 8, v12, v13);
                  v36 = (unsigned int)v66;
                }
              }
              v65[v36] = v35;
              LODWORD(v66) = v66 + 1;
              v37 = *v53;
              if ( (*(_BYTE *)(v59 + 23) & 0x40) != 0 )
                v38 = *(_QWORD *)(v59 - 8);
              else
                v38 = v59 - 24LL * (*(_DWORD *)(v59 + 20) & 0xFFFFFFF);
              v39 = *(_QWORD *)(v38 + 8 * v34);
              v40 = sub_22077B0(32);
              if ( v40 )
                *(_QWORD *)v40 = 0;
              *(_QWORD *)(v40 + 8) = v39;
              *(_QWORD *)(v40 + 16) = a3;
              ++v32;
              *(_DWORD *)(v40 + 24) = v8;
              sub_1C9EE80((_QWORD *)(v37 + 120), 0, (_QWORD *)(v40 + 8), v39, v40);
              v31 = v59;
              v33 = *(_DWORD *)(v59 + 20) & 0xFFFFFFF;
              if ( v33 <= (unsigned int)v32 )
              {
                v7 = v53;
                LODWORD(v15) = v66;
                goto LABEL_18;
              }
            }
          }
LABEL_57:
          LODWORD(v15) = v66;
          goto LABEL_18;
        }
        v59 = 0;
        if ( v18 == 78 )
        {
          v51 = *(_QWORD *)(v17 - 24);
          if ( !*(_BYTE *)(v51 + 16) && (*(_BYTE *)(v51 + 33) & 0x20) != 0 && *(_DWORD *)(v51 + 36) == 3660 )
          {
            v60 = *(__int64 **)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
            sub_12A9700((__int64)&v65, &v60);
            goto LABEL_53;
          }
        }
LABEL_18:
        if ( !(_DWORD)v15 )
          goto LABEL_27;
LABEL_19:
        v14 = v65;
      }
      v41 = *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
      if ( (unsigned int)v15 >= HIDWORD(v66) )
      {
        v56 = *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
        sub_16CD150((__int64)&v65, v67, 0, 8, v41, v13);
        v14 = v65;
        v15 = (unsigned int)v66;
        v41 = v56;
      }
      v14[v15] = v41;
      LODWORD(v66) = v66 + 1;
LABEL_53:
      v42 = *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
      goto LABEL_54;
    }
  }
}
