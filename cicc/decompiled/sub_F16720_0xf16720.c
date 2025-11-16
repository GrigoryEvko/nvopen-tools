// Function: sub_F16720
// Address: 0xf16720
//
unsigned __int8 *__fastcall sub_F16720(__int64 *a1, unsigned __int8 *a2)
{
  unsigned int v4; // ebx
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int8 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  __int64 *v19; // rcx
  char v20; // dl
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // rbx
  __int64 v38; // r13
  __int64 v39; // r13
  __int64 v40; // rbx
  __int64 v41; // r13
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rdx
  int v45; // ecx
  int v46; // eax
  _QWORD *v47; // rdi
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rbx
  __int64 v52; // r13
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v56; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v57; // [rsp+18h] [rbp-B8h]
  unsigned int v58; // [rsp+18h] [rbp-B8h]
  char v59; // [rsp+2Fh] [rbp-A1h] BYREF
  __int64 v60; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-98h]
  _DWORD v62[8]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v63; // [rsp+60h] [rbp-70h]
  _QWORD v64[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v65; // [rsp+90h] [rbp-40h]

  v4 = *a2 - 29;
  if ( *a2 == 58 )
  {
    if ( (a2[1] & 2) == 0 )
      return 0;
LABEL_13:
    v5 = 0;
    v6 = 1;
    if ( (a2[7] & 0x40) == 0 )
      goto LABEL_6;
LABEL_14:
    v7 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    goto LABEL_7;
  }
  if ( v4 > 0x1D )
  {
    if ( *a2 != 82 || sub_B532B0(*((_WORD *)a2 + 1) & 0x3F) )
      return 0;
    goto LABEL_13;
  }
  if ( *a2 == 42 )
    goto LABEL_13;
  if ( *a2 != 44 )
    return 0;
  v5 = 32;
  v6 = 0;
  if ( (a2[7] & 0x40) != 0 )
    goto LABEL_14;
LABEL_6:
  v7 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
LABEL_7:
  v8 = *(_QWORD *)&v7[v5];
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 )
    return 0;
  v10 = *(_QWORD *)(v9 + 8);
  if ( v10 )
    return 0;
  if ( *(_BYTE *)v8 != 85 )
    return 0;
  v12 = *(_QWORD *)(v8 - 32);
  if ( !v12 )
    return 0;
  if ( *(_BYTE *)v12 )
    return 0;
  if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v8 + 80) )
    return 0;
  if ( *(_DWORD *)(v12 + 36) != 66 )
    return 0;
  v13 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
  if ( !v13 )
    return 0;
  v57 = *(unsigned __int8 **)&v7[32 * v6];
  if ( *v57 <= 0x15u && *v57 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v57) )
  {
    v14 = *(_QWORD *)(v13 + 8);
    v15 = sub_BCB060(v14);
    v56 = (unsigned __int8 *)sub_AD64C0(v14, v15, 0);
    if ( v4 != 53
      || (*((_WORD *)a2 + 1) & 0x3Fu) - 32 <= 1
      || (v35 = sub_9719A0(0x22u, v57, (__int64)v56, a1[11], 0, 0)) != 0 && sub_AD7890(v35, (__int64)v57, v33, v34, v36) )
    {
      v16 = *(_QWORD *)(v13 + 16);
      v59 = 0;
      v17 = 0;
      if ( v16 )
        v17 = *(_QWORD *)(v16 + 8) == 0;
      if ( sub_F13D80(a1, v13, v17, 0, &v59, 0) && v59 )
      {
        v18 = *(_QWORD *)(v13 + 16);
        v19 = (__int64 *)a1[4];
        v20 = 0;
        if ( v18 )
          v20 = *(_QWORD *)(v18 + 8) == 0;
        LOBYTE(v64[0]) = 0;
        v21 = sub_F13D80(a1, v13, v20, v19, v64, 0);
        v22 = a1[4];
        v62[1] = 0;
        v65 = 257;
        v60 = v21;
        v55 = sub_B35180(v22, v14, 0x42u, (__int64)&v60, 1u, v62[0], (__int64)v64);
        if ( v4 == 15 )
        {
          v37 = a1[4];
          v63 = 257;
          v38 = sub_AD57F0((__int64)v57, v56, 0, 0);
          v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v37 + 80)
                                                                                             + 32LL))(
                  *(_QWORD *)(v37 + 80),
                  13,
                  v55,
                  v38,
                  0,
                  0);
          if ( !v25 )
          {
            v65 = 257;
            v25 = sub_B504D0(13, v55, v38, (__int64)v64, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v37 + 88) + 16LL))(
              *(_QWORD *)(v37 + 88),
              v25,
              v62,
              *(_QWORD *)(v37 + 56),
              *(_QWORD *)(v37 + 64));
            v39 = 16LL * *(unsigned int *)(v37 + 8);
            v40 = *(_QWORD *)v37;
            v41 = v40 + v39;
            while ( v41 != v40 )
            {
              v42 = *(_QWORD *)(v40 + 8);
              v43 = *(_DWORD *)v40;
              v40 += 16;
              sub_B99FD0(v25, v43, v42);
            }
          }
          return sub_F162A0((__int64)a1, (__int64)a2, v25);
        }
        if ( v4 <= 0xF )
        {
          if ( v4 != 13 )
            goto LABEL_60;
        }
        else if ( v4 != 29 )
        {
          if ( v4 == 53 )
          {
            v63 = 257;
            v23 = a1[4];
            v24 = sub_AD57F0((__int64)v56, v57, 0, 0);
            v58 = sub_B52F50(*((_WORD *)a2 + 1) & 0x3F);
            v25 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v23 + 80) + 56LL))(
                    *(_QWORD *)(v23 + 80),
                    v58,
                    v55,
                    v24);
            if ( !v25 )
            {
              v65 = 257;
              v25 = (__int64)sub_BD2C40(72, unk_3F10FD0);
              if ( v25 )
              {
                v44 = *(_QWORD *)(v55 + 8);
                v45 = *(unsigned __int8 *)(v44 + 8);
                if ( (unsigned int)(v45 - 17) > 1 )
                {
                  v49 = sub_BCB2A0(*(_QWORD **)v44);
                }
                else
                {
                  v46 = *(_DWORD *)(v44 + 32);
                  v47 = *(_QWORD **)v44;
                  BYTE4(v61) = (_BYTE)v45 == 18;
                  LODWORD(v61) = v46;
                  v48 = (__int64 *)sub_BCB2A0(v47);
                  v49 = sub_BCE1B0(v48, v61);
                }
                sub_B523C0(v25, v49, 53, v58, v55, v24, (__int64)v64, 0, 0, 0);
              }
              (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v23 + 88) + 16LL))(
                *(_QWORD *)(v23 + 88),
                v25,
                v62,
                *(_QWORD *)(v23 + 56),
                *(_QWORD *)(v23 + 64));
              v50 = 16LL * *(unsigned int *)(v23 + 8);
              v51 = *(_QWORD *)v23;
              v52 = v51 + v50;
              while ( v52 != v51 )
              {
                v53 = *(_QWORD *)(v51 + 8);
                v54 = *(_DWORD *)v51;
                v51 += 16;
                sub_B99FD0(v25, v54, v53);
              }
            }
            return sub_F162A0((__int64)a1, (__int64)a2, v25);
          }
LABEL_60:
          BUG();
        }
        v26 = a1[4];
        v63 = 257;
        v27 = sub_AD57C0((__int64)v57, v56, 0, 0);
        v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v26 + 80)
                                                                                           + 32LL))(
                *(_QWORD *)(v26 + 80),
                15,
                v27,
                v55,
                0,
                0);
        if ( !v25 )
        {
          v65 = 257;
          v25 = sub_B504D0(15, v27, v55, (__int64)v64, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v26 + 88) + 16LL))(
            *(_QWORD *)(v26 + 88),
            v25,
            v62,
            *(_QWORD *)(v26 + 56),
            *(_QWORD *)(v26 + 64));
          v28 = 16LL * *(unsigned int *)(v26 + 8);
          v29 = *(_QWORD *)v26;
          v30 = v29 + v28;
          while ( v30 != v29 )
          {
            v31 = *(_QWORD *)(v29 + 8);
            v32 = *(_DWORD *)v29;
            v29 += 16;
            sub_B99FD0(v25, v32, v31);
          }
        }
        return sub_F162A0((__int64)a1, (__int64)a2, v25);
      }
    }
    return 0;
  }
  return (unsigned __int8 *)v10;
}
