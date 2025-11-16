// Function: sub_10D1DC0
// Address: 0x10d1dc0
//
__int64 __fastcall sub_10D1DC0(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v4; // eax
  __int64 v5; // r14
  unsigned int v6; // r12d
  _BYTE **v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // r14
  int v10; // eax
  unsigned __int8 *v11; // rdx
  _BYTE *v12; // rdi
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  int v16; // eax
  unsigned __int8 *v17; // rdx
  _BYTE *v18; // rdi
  unsigned __int8 *v19; // r14
  _BYTE *v20; // r14
  char v21; // al
  __int64 v22; // rdx
  unsigned __int8 *v23; // r12
  char v24; // r15
  int v25; // r12d
  unsigned __int8 v26; // al
  __int64 v27; // r14
  __int64 v28; // r13
  __int64 v29; // rdi
  __int64 v30; // r12
  __int64 v31; // rax
  unsigned __int8 *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // r14
  _BYTE **v35; // rdx
  _BYTE *v36; // rdi
  __int64 v37; // r14
  _BYTE *v38; // r14
  char v39; // al
  __int64 v40; // r14
  _BYTE *v41; // rax
  unsigned int v42; // r14d
  _BYTE *v43; // rax
  __int64 v44; // rdx
  int v45; // r13d
  __int64 v46; // r13
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // [rsp+0h] [rbp-120h]
  __int64 v50; // [rsp+0h] [rbp-120h]
  unsigned int v51; // [rsp+Ch] [rbp-114h]
  int v52; // [rsp+Ch] [rbp-114h]
  __int64 v53; // [rsp+10h] [rbp-110h] BYREF
  _BYTE *v54; // [rsp+18h] [rbp-108h] BYREF
  __int64 v55; // [rsp+20h] [rbp-100h] BYREF
  __int64 v56; // [rsp+28h] [rbp-F8h] BYREF
  _BYTE v57[32]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v58; // [rsp+50h] [rbp-D0h]
  _BYTE v59[32]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v60; // [rsp+80h] [rbp-A0h]
  __int64 *v61; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v62; // [rsp+98h] [rbp-88h]
  __int64 *v63; // [rsp+A0h] [rbp-80h]
  __int64 *v64; // [rsp+A8h] [rbp-78h]
  __int16 v65; // [rsp+B0h] [rbp-70h]
  __int64 *v66; // [rsp+B8h] [rbp-68h]
  _QWORD *v67[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 *v68; // [rsp+D0h] [rbp-50h]
  _QWORD *v69[9]; // [rsp+D8h] [rbp-48h] BYREF

  v4 = sub_BCB060(*((_QWORD *)a2 + 1));
  v5 = *((_QWORD *)a2 - 8);
  v6 = v4;
  v64 = &v55;
  v61 = (__int64 *)&v54;
  v63 = &v53;
  v66 = &v53;
  v67[0] = &v56;
  v68 = &v53;
  v69[0] = &v56;
  if ( (unsigned __int8)(*(_BYTE *)v5 - 54) <= 2u )
  {
    v7 = (*(_BYTE *)(v5 + 7) & 0x40) != 0
       ? *(_BYTE ***)(v5 - 8)
       : (_BYTE **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    v8 = *v7;
    if ( **v7 <= 0x15u )
    {
      v54 = *v7;
      if ( *v8 > 0x15u || *v8 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v8) )
      {
        v14 = (*(_BYTE *)(v5 + 7) & 0x40) != 0 ? *(_QWORD *)(v5 - 8) : v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
        v15 = *(_QWORD *)(v14 + 32);
        if ( v15 )
        {
          *v63 = v15;
          v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
          v16 = *v9;
          if ( (unsigned __int8)v16 <= 0x1Cu || (unsigned int)(v16 - 54) > 2 )
            return 0;
          if ( (v9[7] & 0x40) != 0 )
            v17 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
          else
            v17 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
          v18 = *(_BYTE **)v17;
          if ( **(_BYTE **)v17 > 0x15u )
            goto LABEL_9;
          *v64 = (__int64)v18;
          if ( *v18 > 0x15u || *v18 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v18) )
          {
            if ( (v9[7] & 0x40) != 0 )
              v19 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
            else
              v19 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
            v20 = (_BYTE *)*((_QWORD *)v19 + 4);
            v21 = *v20;
            if ( *v20 != 42 )
            {
LABEL_32:
              if ( v21 == 58
                && (v20[1] & 2) != 0
                && *((_QWORD *)v20 - 8) == *v68
                && (unsigned __int8)sub_F11D70(v69, *((_BYTE **)v20 - 4)) )
              {
                goto LABEL_36;
              }
              goto LABEL_7;
            }
            if ( *((_QWORD *)v20 - 8) == *v66 )
            {
              if ( (unsigned __int8)sub_F11D70(v67, *((_BYTE **)v20 - 4)) )
                goto LABEL_36;
              v21 = *v20;
              goto LABEL_32;
            }
          }
        }
      }
    }
  }
LABEL_7:
  v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v10 = *v9;
  if ( (unsigned __int8)v10 <= 0x1Cu || (unsigned int)(v10 - 54) > 2 )
    return 0;
LABEL_9:
  if ( (v9[7] & 0x40) != 0 )
    v11 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
  else
    v11 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
  v12 = *(_BYTE **)v11;
  if ( **(_BYTE **)v11 > 0x15u )
    return 0;
  *v61 = (__int64)v12;
  if ( *v12 <= 0x15u && (*v12 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v12)) )
    return 0;
  v32 = (v9[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v9 - 1) : &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
  v33 = *((_QWORD *)v32 + 4);
  if ( !v33 )
    return 0;
  *v63 = v33;
  v34 = *((_QWORD *)a2 - 8);
  if ( (unsigned __int8)(*(_BYTE *)v34 - 54) > 2u )
    return 0;
  v35 = (*(_BYTE *)(v34 + 7) & 0x40) != 0
      ? *(_BYTE ***)(v34 - 8)
      : (_BYTE **)(v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF));
  v36 = *v35;
  if ( **v35 > 0x15u )
    return 0;
  *v64 = (__int64)v36;
  if ( *v36 <= 0x15u && (*v36 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v36)) )
    return 0;
  if ( (*(_BYTE *)(v34 + 7) & 0x40) != 0 )
    v37 = *(_QWORD *)(v34 - 8);
  else
    v37 = v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF);
  v38 = *(_BYTE **)(v37 + 32);
  v39 = *v38;
  if ( *v38 != 42 )
  {
LABEL_63:
    if ( v39 != 58
      || (v38[1] & 2) == 0
      || *((_QWORD *)v38 - 8) != *v68
      || !(unsigned __int8)sub_F11D70(v69, *((_BYTE **)v38 - 4)) )
    {
      return 0;
    }
    goto LABEL_36;
  }
  if ( *((_QWORD *)v38 - 8) != *v66 )
    return 0;
  if ( !(unsigned __int8)sub_F11D70(v67, *((_BYTE **)v38 - 4)) )
  {
    v39 = *v38;
    goto LABEL_63;
  }
LABEL_36:
  v62 = v6;
  if ( v6 > 0x40 )
    sub_C43690((__int64)&v61, v6, 0);
  else
    v61 = (__int64 *)v6;
  v23 = (unsigned __int8 *)v56;
  if ( *(_BYTE *)v56 == 17 )
  {
    v24 = sub_B532C0(v56 + 24, &v61, 36);
  }
  else
  {
    v40 = *(_QWORD *)(v56 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 > 1 )
      goto LABEL_71;
    v41 = sub_AD7630(v56, 0, v22);
    if ( !v41 || *v41 != 17 )
    {
      if ( *(_BYTE *)(v40 + 8) == 17 )
      {
        v52 = *(_DWORD *)(v40 + 32);
        if ( v52 )
        {
          v24 = 0;
          v42 = 0;
          while ( 1 )
          {
            v43 = (_BYTE *)sub_AD69F0(v23, v42);
            if ( !v43 )
              break;
            if ( *v43 != 13 )
            {
              if ( *v43 != 17 )
                break;
              v24 = sub_B532C0((__int64)(v43 + 24), &v61, 36);
              if ( !v24 )
                break;
            }
            if ( v52 == ++v42 )
              goto LABEL_40;
          }
        }
      }
      goto LABEL_71;
    }
    v24 = sub_B532C0((__int64)(v41 + 24), &v61, 36);
  }
LABEL_40:
  if ( !v24 )
  {
LABEL_71:
    if ( v62 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
    return 0;
  }
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  v25 = **((unsigned __int8 **)a2 - 8);
  result = 0;
  if ( (unsigned __int8)v25 > 0x1Cu )
  {
    v26 = **((_BYTE **)a2 - 4);
    if ( v26 > 0x1Cu && (_BYTE)v25 == v26 )
    {
      v51 = v25 - 29;
      if ( *a2 != 42 || v25 == 54 )
      {
        v27 = *(_QWORD *)(a1 + 32);
        v60 = 257;
        v28 = v55;
        v29 = *(_QWORD *)(v27 + 80);
        v58 = 257;
        v49 = v56;
        v30 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v29 + 16LL))(v29, v51, v55, v56);
        if ( !v30 )
        {
          v65 = 257;
          v30 = sub_B504D0(v51, v28, v49, (__int64)&v61, 0, 0);
          if ( (unsigned __int8)sub_920620(v30) )
          {
            v44 = *(_QWORD *)(v27 + 96);
            v45 = *(_DWORD *)(v27 + 104);
            if ( v44 )
              sub_B99FD0(v30, 3u, v44);
            sub_B45150(v30, v45);
          }
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v27 + 88) + 16LL))(
            *(_QWORD *)(v27 + 88),
            v30,
            v57,
            *(_QWORD *)(v27 + 56),
            *(_QWORD *)(v27 + 64));
          v46 = *(_QWORD *)v27;
          v50 = *(_QWORD *)v27 + 16LL * *(unsigned int *)(v27 + 8);
          if ( *(_QWORD *)v27 != v50 )
          {
            do
            {
              v47 = *(_QWORD *)(v46 + 8);
              v48 = *(_DWORD *)v46;
              v46 += 16;
              sub_B99FD0(v30, v48, v47);
            }
            while ( v50 != v46 );
          }
        }
        v31 = sub_10BBE20((__int64 *)v27, (unsigned int)*a2 - 29, (__int64)v54, v30, (int)v61, 0, (__int64)v59, 0);
        v65 = 257;
        return sub_B504D0(v51, v31, v53, (__int64)&v61, 0, 0);
      }
    }
    return 0;
  }
  return result;
}
