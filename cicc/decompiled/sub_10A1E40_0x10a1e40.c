// Function: sub_10A1E40
// Address: 0x10a1e40
//
__int64 __fastcall sub_10A1E40(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // r12
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // r13
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rax
  int v21; // r14d
  unsigned int v22; // eax
  __int64 *v23; // rbx
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // r12
  __int64 v27; // r15
  int v28; // edx
  __int64 v29; // r10
  __int64 v30; // r14
  __int64 v31; // r15
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rsi
  int v35; // edi
  __int64 v36; // rax
  unsigned int v37; // edi
  _QWORD *v38; // rax
  __int64 v39; // r15
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 v42; // rdx
  unsigned int v43; // esi
  _BYTE *v44; // r14
  __int64 v45; // r15
  __int64 *v46; // r13
  unsigned int v47; // eax
  _BYTE *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 v52; // r14
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // rdx
  int v56; // r14d
  __int64 *v57; // rax
  int v58; // [rsp+4h] [rbp-10Ch]
  __int64 v59; // [rsp+10h] [rbp-100h]
  int v60; // [rsp+18h] [rbp-F8h]
  int v61; // [rsp+1Ch] [rbp-F4h]
  __int64 v62; // [rsp+20h] [rbp-F0h]
  __int64 v63; // [rsp+28h] [rbp-E8h]
  __int64 v64; // [rsp+38h] [rbp-D8h]
  __int64 v65[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v66[4]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v67; // [rsp+70h] [rbp-A0h]
  _BYTE v68[32]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-70h]
  _BYTE v70[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v71; // [rsp+D0h] [rbp-40h]

  v5 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 47 )
  {
    v44 = *(_BYTE **)(a2 - 64);
    if ( v44 )
    {
      v45 = *(_QWORD *)(a2 - 32);
      if ( v45 )
      {
        v46 = *(__int64 **)(a1 + 32);
        v71 = 257;
        BYTE4(v66[0]) = 1;
        LODWORD(v66[0]) = sub_B45210(a3);
        v69 = 257;
        v47 = sub_B45210(a3);
        v48 = (_BYTE *)sub_109D090(v46, v45, v47, 1, (__int64)v68, 0);
        return sub_A826E0((unsigned int **)v46, v44, v48, v66[0], (__int64)v70, 0);
      }
    }
    return 0;
  }
  if ( v5 != 50 )
  {
    if ( v5 == 85 )
    {
      v20 = *(_QWORD *)(a2 - 32);
      if ( v20 )
      {
        if ( !*(_BYTE *)v20
          && *(_QWORD *)(v20 + 24) == *(_QWORD *)(a2 + 80)
          && (*(_BYTE *)(v20 + 33) & 0x20) != 0
          && *(_DWORD *)(v20 + 36) == 209 )
        {
          v21 = sub_B45210(a2);
          v22 = sub_B45210(a3);
          v23 = *(__int64 **)(a1 + 32);
          v69 = 257;
          v61 = v21 | v22;
          v24 = v21 | v22;
          v25 = *(_DWORD *)(a2 + 4);
          v67 = 257;
          v26 = *(_QWORD *)(a2 - 32LL * (v25 & 0x7FFFFFF));
          v27 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v23[10] + 48LL))(
                  v23[10],
                  12,
                  v26,
                  v24);
          if ( !v27 )
          {
            v71 = 257;
            v49 = sub_B50340(12, v26, (__int64)v70, 0, 0);
            v50 = v23[12];
            v27 = v49;
            if ( v50 )
              sub_B99FD0(v49, 3u, v50);
            sub_B45150(v27, v61);
            (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
              v23[11],
              v27,
              v66,
              v23[7],
              v23[8]);
            v51 = *v23;
            v52 = *v23 + 16LL * *((unsigned int *)v23 + 2);
            if ( *v23 != v52 )
            {
              do
              {
                v53 = *(_QWORD *)(v51 + 8);
                v54 = *(_DWORD *)v51;
                v51 += 16;
                sub_B99FD0(v27, v54, v53);
              }
              while ( v52 != v51 );
            }
          }
          v28 = *(_DWORD *)(a2 + 4);
          v29 = *(_QWORD *)(a2 - 32);
          v65[0] = v27;
          v65[1] = *(_QWORD *)(a2 + 32 * (1LL - (v28 & 0x7FFFFFF)));
          if ( v29 )
          {
            if ( !*(_BYTE *)v29 )
            {
              v30 = *(_QWORD *)(a2 + 80);
              if ( *(_QWORD *)(v29 + 24) == v30 )
              {
LABEL_26:
                v71 = 257;
                v31 = v23[15];
                v32 = v23[14];
                v33 = v32 + 56 * v31;
                if ( v33 == v32 )
                {
                  v58 = 3;
                  v37 = 3;
                }
                else
                {
                  v34 = v23[14];
                  v35 = 0;
                  do
                  {
                    v36 = *(_QWORD *)(v34 + 40) - *(_QWORD *)(v34 + 32);
                    v34 += 56;
                    v35 += v36 >> 3;
                  }
                  while ( v33 != v34 );
                  v37 = v35 + 3;
                  v58 = v37 & 0x7FFFFFF;
                }
                v59 = v23[14];
                v62 = v29;
                LOBYTE(v60) = 16 * (_DWORD)v31 != 0;
                v38 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v31) << 32) | v37);
                v6 = (__int64)v38;
                if ( v38 )
                {
                  v63 = v31;
                  v39 = (__int64)v38;
                  sub_B44260((__int64)v38, **(_QWORD **)(v30 + 16), 56, v58 | (v60 << 28), 0, 0);
                  *(_QWORD *)(v6 + 72) = 0;
                  sub_B4A290(v6, v30, v62, v65, 2, (__int64)v70, v59, v63);
                }
                else
                {
                  v39 = 0;
                }
                if ( *((_BYTE *)v23 + 108) )
                {
                  v57 = (__int64 *)sub_BD5C60(v39);
                  *(_QWORD *)(v6 + 72) = sub_A7A090((__int64 *)(v6 + 72), v57, -1, 72);
                }
                if ( (unsigned __int8)sub_920620(v39) )
                {
                  v55 = v23[12];
                  v56 = *((_DWORD *)v23 + 26);
                  if ( v55 )
                    sub_B99FD0(v6, 3u, v55);
                  sub_B45150(v6, v56);
                }
                (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
                  v23[11],
                  v6,
                  v68,
                  v23[7],
                  v23[8]);
                v40 = *v23;
                v41 = *v23 + 16LL * *((unsigned int *)v23 + 2);
                while ( v41 != v40 )
                {
                  v42 = *(_QWORD *)(v40 + 8);
                  v43 = *(_DWORD *)v40;
                  v40 += 16;
                  sub_B99FD0(v6, v43, v42);
                }
                sub_B45150(v39, v61);
                sub_B47C00(v39, a2, 0, 0);
                return v6;
              }
            }
            v29 = 0;
          }
          v30 = 0;
          goto LABEL_26;
        }
      }
    }
    return 0;
  }
  v8 = *(_QWORD *)(a2 - 64);
  if ( !v8 )
    return 0;
  v9 = *(_QWORD *)(a2 - 32);
  if ( !v9 )
    return 0;
  v10 = *(_QWORD *)(a1 + 32);
  v69 = 257;
  v67 = 257;
  LODWORD(v64) = sub_B45210(a3);
  v11 = sub_B45210(a3);
  v12 = sub_109D090((__int64 *)v10, v8, v11, 1, (__int64)v66, 0);
  BYTE4(v64) = 1;
  v13 = v12;
  v65[0] = v64;
  if ( *(_BYTE *)(v10 + 108) )
    return sub_B35400(v10, 0x69u, v12, v9, v64, (__int64)v68, 0, 0, 0);
  v6 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v10 + 80) + 40LL))(
         *(_QWORD *)(v10 + 80),
         21,
         v12,
         v9,
         (unsigned int)v64);
  if ( !v6 )
  {
    v71 = 257;
    v14 = sub_B504D0(21, v13, v9, (__int64)v70, 0, 0);
    v15 = *(_QWORD *)(v10 + 96);
    v6 = v14;
    if ( v15 )
      sub_B99FD0(v14, 3u, v15);
    sub_B45150(v6, v64);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v10 + 88) + 16LL))(
      *(_QWORD *)(v10 + 88),
      v6,
      v68,
      *(_QWORD *)(v10 + 56),
      *(_QWORD *)(v10 + 64));
    v16 = *(_QWORD *)v10;
    v17 = *(_QWORD *)v10 + 16LL * *(unsigned int *)(v10 + 8);
    while ( v17 != v16 )
    {
      v18 = *(_QWORD *)(v16 + 8);
      v19 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v6, v19, v18);
    }
  }
  return v6;
}
