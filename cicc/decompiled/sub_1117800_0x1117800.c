// Function: sub_1117800
// Address: 0x1117800
//
__int64 __fastcall sub_1117800(
        const __m128i *a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int8 *a6,
        unsigned __int8 **a7,
        __int64 *a8)
{
  _QWORD *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rdi
  _BOOL4 v17; // eax
  __int64 v18; // rdi
  bool v19; // al
  int v20; // eax
  __int64 v21; // rbx
  unsigned __int8 *v22; // r9
  unsigned __int8 *v23; // rdi
  __int64 result; // rax
  __int64 v25; // rbx
  unsigned __int8 *v26; // r9
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  _BYTE *v29; // rax
  bool v30; // dl
  char v31; // al
  __int64 v32; // r9
  __int64 v33; // rdx
  int v34; // r12d
  __int64 v35; // r12
  __int64 i; // rbx
  __int64 v37; // rdx
  unsigned int v38; // esi
  char v39; // al
  __int64 v40; // r9
  __int64 v41; // rdx
  int v42; // r12d
  __int64 v43; // r12
  __int64 v44; // rbx
  __int64 v45; // rdx
  unsigned int v46; // esi
  _BYTE *v47; // rax
  unsigned int v48; // ecx
  __int64 v49; // rax
  unsigned int v50; // ecx
  unsigned int v51; // ecx
  _BYTE *v52; // rax
  unsigned int v53; // ecx
  bool v54; // al
  int v55; // [rsp+0h] [rbp-C0h]
  int v56; // [rsp+0h] [rbp-C0h]
  bool v57; // [rsp+4h] [rbp-BCh]
  bool v58; // [rsp+4h] [rbp-BCh]
  unsigned int v59; // [rsp+4h] [rbp-BCh]
  bool v60; // [rsp+4h] [rbp-BCh]
  int v61; // [rsp+8h] [rbp-B8h]
  __int64 v62; // [rsp+8h] [rbp-B8h]
  __int64 v63; // [rsp+8h] [rbp-B8h]
  __int64 v64; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v65; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v66; // [rsp+8h] [rbp-B8h]
  int v67; // [rsp+8h] [rbp-B8h]
  __int64 v68; // [rsp+8h] [rbp-B8h]
  unsigned int v69; // [rsp+8h] [rbp-B8h]
  int v70; // [rsp+8h] [rbp-B8h]
  unsigned int v71; // [rsp+8h] [rbp-B8h]
  __int64 v73; // [rsp+10h] [rbp-B0h]
  __int64 v74; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v75; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v76; // [rsp+10h] [rbp-B0h]
  __int64 v77; // [rsp+18h] [rbp-A8h]
  __int64 v78; // [rsp+28h] [rbp-98h]
  _BYTE v79[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v80; // [rsp+50h] [rbp-70h]
  _BYTE v81[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v82; // [rsp+80h] [rbp-40h]

  if ( sub_B46D50(a6) && *(_BYTE *)a4 <= 0x15u && *(_BYTE *)a5 > 0x15u )
  {
    v28 = a4;
    a4 = a5;
    a5 = v28;
  }
  sub_D5F1F0(a1[2].m128i_i64[0], (__int64)a6);
  v12 = (_QWORD *)sub_BD5C60(a4);
  v13 = (__int64 *)sub_BCB2A0(v12);
  v14 = *(_QWORD *)(a4 + 8);
  v77 = (__int64)v13;
  v15 = *(unsigned __int8 *)(v14 + 8);
  if ( (unsigned int)(v15 - 17) <= 1 )
  {
    BYTE4(v78) = (_BYTE)v15 == 18;
    LODWORD(v78) = *(_DWORD *)(v14 + 32);
    v77 = sub_BCE1B0(v13, v78);
  }
  if ( a2 == 17 )
  {
    v16 = *(_QWORD *)(a5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    LOBYTE(v17) = a3 & sub_BCAC40(v16, 1);
    if ( !v17 )
    {
      if ( *(_BYTE *)a5 == 17 )
      {
        if ( *(_DWORD *)(a5 + 32) > 0x40u )
        {
          v61 = *(_DWORD *)(a5 + 32);
          v18 = a5 + 24;
LABEL_12:
          v19 = v61 - 1 == (unsigned int)sub_C444A0(v18);
          goto LABEL_13;
        }
        v19 = *(_QWORD *)(a5 + 24) == 1;
        goto LABEL_13;
      }
      v57 = v17;
      v62 = *(_QWORD *)(a5 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v62 + 8) - 17 <= 1 && *(_BYTE *)a5 <= 0x15u )
      {
        v29 = sub_AD7630(a5, 0, v17);
        v30 = v57;
        if ( v29 && *v29 == 17 )
        {
          if ( *((_DWORD *)v29 + 8) > 0x40u )
          {
            v61 = *((_DWORD *)v29 + 8);
            v18 = (__int64)(v29 + 24);
            goto LABEL_12;
          }
          v19 = *((_QWORD *)v29 + 3) == 1;
LABEL_13:
          if ( !v19 )
            goto LABEL_14;
LABEL_28:
          *a7 = (unsigned __int8 *)a4;
          *a8 = sub_AD6450(v77);
          return 1;
        }
        if ( *(_BYTE *)(v62 + 8) == 17 )
        {
          v55 = *(_DWORD *)(v62 + 32);
          if ( v55 )
          {
            v48 = 0;
            while ( 1 )
            {
              v58 = v30;
              v69 = v48;
              v49 = sub_AD69F0((unsigned __int8 *)a5, v48);
              if ( !v49 )
                break;
              v50 = v69;
              v30 = v58;
              if ( *(_BYTE *)v49 != 13 )
              {
                if ( *(_BYTE *)v49 != 17 )
                  break;
                if ( *(_DWORD *)(v49 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v49 + 24) != 1 )
                    break;
                  v30 = 1;
                }
                else
                {
                  v70 = *(_DWORD *)(v49 + 32);
                  v59 = v50;
                  if ( (unsigned int)sub_C444A0(v49 + 24) != v70 - 1 )
                    break;
                  v50 = v59;
                  v30 = 1;
                }
              }
              v48 = v50 + 1;
              if ( v55 == v48 )
                goto LABEL_63;
            }
          }
        }
      }
    }
  }
  else
  {
    if ( a2 > 0x11 || (a2 & 0xFFFFFFFD) != 0xD )
      goto LABEL_89;
    if ( *(_BYTE *)a5 > 0x15u )
      goto LABEL_14;
    if ( sub_AC30F0(a5) )
      goto LABEL_28;
    if ( *(_BYTE *)a5 == 17 )
    {
      if ( *(_DWORD *)(a5 + 32) <= 0x40u )
      {
        v19 = *(_QWORD *)(a5 + 24) == 0;
      }
      else
      {
        v67 = *(_DWORD *)(a5 + 32);
        v19 = v67 == (unsigned int)sub_C444A0(a5 + 24);
      }
      goto LABEL_13;
    }
    v68 = *(_QWORD *)(a5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v68 + 8) - 17 <= 1 )
    {
      v47 = sub_AD7630(a5, 0, v27);
      v30 = 0;
      if ( v47 && *v47 == 17 )
      {
        v30 = sub_9867B0((__int64)(v47 + 24));
LABEL_63:
        if ( v30 )
          goto LABEL_28;
      }
      else if ( *(_BYTE *)(v68 + 8) == 17 )
      {
        v56 = *(_DWORD *)(v68 + 32);
        if ( v56 )
        {
          v51 = 0;
          while ( 1 )
          {
            v60 = v30;
            v71 = v51;
            v52 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a5, v51);
            if ( !v52 )
              break;
            v53 = v71;
            v30 = v60;
            if ( *v52 != 13 )
            {
              if ( *v52 != 17 )
                break;
              v54 = sub_9867B0((__int64)(v52 + 24));
              v53 = v71;
              v30 = v54;
              if ( !v54 )
                break;
            }
            v51 = v53 + 1;
            if ( v56 == v51 )
              goto LABEL_63;
          }
        }
      }
    }
  }
LABEL_14:
  v20 = sub_1117350(a1, a2, a3, a4, a5, (__int64)a6);
  if ( v20 == 2 )
    return 0;
  if ( v20 > 2 )
  {
    if ( v20 == 3 )
    {
      v21 = a1[2].m128i_i64[0];
      v80 = 257;
      v22 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, unsigned __int64))(**(_QWORD **)(v21 + 80) + 16LL))(
                                 *(_QWORD *)(v21 + 80),
                                 a2,
                                 a4,
                                 a5);
      if ( !v22 )
      {
        v82 = 257;
        v63 = sub_B504D0(a2, a4, a5, (__int64)v81, 0, 0);
        v31 = sub_920620(v63);
        v32 = v63;
        if ( v31 )
        {
          v33 = *(_QWORD *)(v21 + 96);
          v34 = *(_DWORD *)(v21 + 104);
          if ( v33 )
          {
            sub_B99FD0(v63, 3u, v33);
            v32 = v63;
          }
          v64 = v32;
          sub_B45150(v32, v34);
          v32 = v64;
        }
        v65 = (unsigned __int8 *)v32;
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v21 + 88) + 16LL))(
          *(_QWORD *)(v21 + 88),
          v32,
          v79,
          *(_QWORD *)(v21 + 56),
          *(_QWORD *)(v21 + 64));
        v35 = *(_QWORD *)v21;
        v22 = v65;
        for ( i = *(_QWORD *)v21 + 16LL * *(unsigned int *)(v21 + 8); i != v35; v22 = v66 )
        {
          v37 = *(_QWORD *)(v35 + 8);
          v38 = *(_DWORD *)v35;
          v35 += 16;
          v66 = v22;
          sub_B99FD0((__int64)v22, v38, v37);
        }
      }
      *a7 = v22;
      sub_BD6B90(v22, a6);
      *a8 = sub_AD6450(v77);
      v23 = *a7;
      result = 1;
      if ( **a7 > 0x1Cu )
      {
        if ( (_BYTE)a3 )
        {
          sub_B44850(v23, 1);
          return a3;
        }
        else
        {
          sub_B447F0(v23, 1);
          return 1;
        }
      }
      return result;
    }
LABEL_89:
    BUG();
  }
  if ( (unsigned int)v20 > 1 )
    goto LABEL_89;
  v25 = a1[2].m128i_i64[0];
  v80 = 257;
  v26 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, unsigned __int64))(**(_QWORD **)(v25 + 80)
                                                                                                + 16LL))(
                             *(_QWORD *)(v25 + 80),
                             a2,
                             a4,
                             a5);
  if ( !v26 )
  {
    v82 = 257;
    v73 = sub_B504D0(a2, a4, a5, (__int64)v81, 0, 0);
    v39 = sub_920620(v73);
    v40 = v73;
    if ( v39 )
    {
      v41 = *(_QWORD *)(v25 + 96);
      v42 = *(_DWORD *)(v25 + 104);
      if ( v41 )
      {
        sub_B99FD0(v73, 3u, v41);
        v40 = v73;
      }
      v74 = v40;
      sub_B45150(v40, v42);
      v40 = v74;
    }
    v75 = (unsigned __int8 *)v40;
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v25 + 88) + 16LL))(
      *(_QWORD *)(v25 + 88),
      v40,
      v79,
      *(_QWORD *)(v25 + 56),
      *(_QWORD *)(v25 + 64));
    v26 = v75;
    v43 = *(_QWORD *)v25 + 16LL * *(unsigned int *)(v25 + 8);
    if ( *(_QWORD *)v25 != v43 )
    {
      v44 = *(_QWORD *)v25;
      do
      {
        v45 = *(_QWORD *)(v44 + 8);
        v46 = *(_DWORD *)v44;
        v44 += 16;
        v76 = v26;
        sub_B99FD0((__int64)v26, v46, v45);
        v26 = v76;
      }
      while ( v43 != v44 );
    }
  }
  *a7 = v26;
  sub_BD6B90(v26, a6);
  *a8 = sub_AD6400(v77);
  return 1;
}
