// Function: sub_18E7670
// Address: 0x18e7670
//
void __fastcall sub_18E7670(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, char a7)
{
  __int64 v7; // r15
  unsigned __int64 v8; // r12
  __int64 *v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // r9
  __int64 v13; // rcx
  unsigned int v14; // eax
  _QWORD *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r8d
  int v20; // r9d
  int v21; // r9d
  unsigned __int64 *v22; // rdx
  int v23; // eax
  char *v24; // rdi
  unsigned __int64 *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // r8d
  int v29; // r9d
  unsigned __int64 *v30; // rax
  char *v31; // rdi
  _QWORD *v32; // r8
  unsigned __int64 v33; // r13
  unsigned __int64 *v34; // rbx
  unsigned __int64 *v35; // r15
  __int64 v36; // r14
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rdx
  unsigned int v42; // eax
  unsigned int v43; // eax
  int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rcx
  int v47; // r8d
  int v48; // r9d
  char *v49; // rdi
  unsigned int v50; // eax
  unsigned int v51; // eax
  unsigned int v52; // eax
  __int64 v53; // r14
  int v54; // r8d
  int v55; // r9d
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rbx
  __int64 v59; // rdx
  __int64 v60; // rcx
  int v61; // r8d
  int v62; // r9d
  int v63; // eax
  int v64; // esi
  unsigned int v65; // eax
  unsigned int v66; // eax
  char v67; // [rsp+0h] [rbp-130h]
  char **v68; // [rsp+8h] [rbp-128h]
  int v69; // [rsp+10h] [rbp-120h]
  __int64 v70; // [rsp+18h] [rbp-118h]
  __int64 v71; // [rsp+20h] [rbp-110h]
  __int64 v72; // [rsp+20h] [rbp-110h]
  __int64 v73; // [rsp+38h] [rbp-F8h]
  __int64 v74; // [rsp+38h] [rbp-F8h]
  __int64 v75; // [rsp+40h] [rbp-F0h]
  __int64 v76; // [rsp+40h] [rbp-F0h]
  __int64 v77; // [rsp+40h] [rbp-F0h]
  __int64 v78; // [rsp+40h] [rbp-F0h]
  _QWORD *v79; // [rsp+48h] [rbp-E8h]
  __int64 v80; // [rsp+48h] [rbp-E8h]
  __int64 *v81; // [rsp+48h] [rbp-E8h]
  __int64 *v82; // [rsp+48h] [rbp-E8h]
  __int64 v83; // [rsp+48h] [rbp-E8h]
  unsigned int v84; // [rsp+48h] [rbp-E8h]
  char v85; // [rsp+5Fh] [rbp-D1h] BYREF
  char *v86; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v87; // [rsp+68h] [rbp-C8h]
  _BYTE v88[128]; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 *v89; // [rsp+F0h] [rbp-40h]
  int v90; // [rsp+F8h] [rbp-38h]

  v70 = a3;
  if ( (__int64)(a2 - a1) > 2560 )
  {
    v7 = a1;
    if ( a3 )
    {
      v8 = a2;
      v68 = (char **)(a1 + 160);
      while ( 1 )
      {
        v9 = *(__int64 **)(v7 + 304);
        --v70;
        v10 = *v9;
        v11 = v7 + 160 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * ((__int64)(v8 - v7) >> 5)) / 2);
        v12 = *(_QWORD *)(v11 + 144);
        v13 = *(_QWORD *)v12;
        if ( *v9 == *(_QWORD *)v12 )
        {
          v72 = *(_QWORD *)v12;
          v74 = *v9;
          v76 = *(_QWORD *)(v11 + 144);
          v81 = *(__int64 **)(v7 + 304);
          v52 = sub_16A9900((__int64)(v9 + 3), (unsigned __int64 *)(v12 + 24));
          v9 = v81;
          v12 = v76;
          v10 = v74;
          v13 = v72;
          v14 = v52 >> 31;
        }
        else
        {
          LOBYTE(v14) = *(_DWORD *)(v10 + 8) >> 8 < *(_DWORD *)(v13 + 8) >> 8;
        }
        v15 = *(_QWORD **)(v8 - 16);
        v16 = *v15;
        if ( (_BYTE)v14 )
        {
          if ( v13 == v16 )
          {
            v77 = v10;
            v82 = v9;
            v63 = sub_16A9900(v12 + 24, v15 + 3);
            v9 = v82;
            v10 = v77;
            if ( v63 < 0 )
            {
LABEL_59:
              v64 = *(_DWORD *)(v7 + 8);
              v86 = v88;
              v87 = 0x800000000LL;
              if ( !v64 )
                goto LABEL_56;
              goto LABEL_60;
            }
          }
          else
          {
            v13 = *(_DWORD *)(v13 + 8) >> 8;
            if ( (unsigned int)v13 < *(_DWORD *)(v16 + 8) >> 8 )
              goto LABEL_59;
          }
          if ( v10 == v16 )
          {
            if ( (int)sub_16A9900((__int64)(v9 + 3), v15 + 3) < 0 )
            {
LABEL_11:
              v13 = *(unsigned int *)(v7 + 8);
              v86 = v88;
              v87 = 0x800000000LL;
              if ( (_DWORD)v13 )
                goto LABEL_49;
              goto LABEL_12;
            }
          }
          else
          {
            v10 = *(_DWORD *)(v10 + 8) >> 8;
            if ( (unsigned int)v10 < *(_DWORD *)(v16 + 8) >> 8 )
              goto LABEL_11;
          }
          v10 = *(unsigned int *)(v7 + 8);
          v86 = v88;
          v87 = 0x800000000LL;
          if ( (_DWORD)v10 )
            goto LABEL_62;
        }
        else
        {
          if ( v10 == v16 )
          {
            v78 = v13;
            v83 = v12;
            v65 = sub_16A9900((__int64)(v9 + 3), v15 + 3);
            v12 = v83;
            v13 = v78;
            v50 = v65 >> 31;
          }
          else
          {
            LOBYTE(v50) = *(_DWORD *)(v10 + 8) >> 8 < *(_DWORD *)(v16 + 8) >> 8;
          }
          v10 = *(unsigned int *)(v7 + 8);
          if ( !(_BYTE)v50 )
          {
            if ( v13 == v16 )
            {
              v84 = *(_DWORD *)(v7 + 8);
              v66 = sub_16A9900(v12 + 24, v15 + 3);
              v10 = v84;
              v51 = v66 >> 31;
            }
            else
            {
              v13 = *(_DWORD *)(v13 + 8) >> 8;
              LOBYTE(v51) = (unsigned int)v13 < *(_DWORD *)(v16 + 8) >> 8;
            }
            if ( (_BYTE)v51 )
            {
              v86 = v88;
              v87 = 0x800000000LL;
              if ( (_DWORD)v10 )
LABEL_49:
                sub_18E63F0((__int64)&v86, (char **)v7, v10, v13, (int)v9, v12);
LABEL_12:
              v89 = *(unsigned __int64 **)(v7 + 144);
              v90 = *(_DWORD *)(v7 + 152);
              sub_18E63F0(v7, (char **)(v8 - 160), v10, v13, (int)v9, v12);
              *(_QWORD *)(v7 + 144) = *(_QWORD *)(v8 - 16);
              *(_DWORD *)(v7 + 152) = *(_DWORD *)(v8 - 8);
              sub_18E63F0(v8 - 160, &v86, v17, v18, v19, v20);
              v22 = v89;
              v23 = v90;
              v24 = v86;
              *(_QWORD *)(v8 - 16) = v89;
              *(_DWORD *)(v8 - 8) = v23;
              if ( v24 != v88 )
              {
                _libc_free((unsigned __int64)v24);
                v22 = *(unsigned __int64 **)(v8 - 16);
              }
              v25 = *(unsigned __int64 **)(v7 + 304);
              goto LABEL_21;
            }
            v86 = v88;
            v87 = 0x800000000LL;
            if ( !(_DWORD)v10 )
            {
LABEL_56:
              v89 = *(unsigned __int64 **)(v7 + 144);
              v90 = *(_DWORD *)(v7 + 152);
              sub_18E63F0(v7, (char **)v11, v10, v13, (int)v9, v12);
              *(_QWORD *)(v7 + 144) = *(_QWORD *)(v11 + 144);
              *(_DWORD *)(v7 + 152) = *(_DWORD *)(v11 + 152);
              sub_18E63F0(v11, &v86, v59, v60, v61, v62);
              v31 = v86;
              *(_QWORD *)(v11 + 144) = v89;
              *(_DWORD *)(v11 + 152) = v90;
              if ( v31 != v88 )
LABEL_18:
                _libc_free((unsigned __int64)v31);
              v25 = *(unsigned __int64 **)(v7 + 304);
              goto LABEL_20;
            }
LABEL_60:
            sub_18E63F0((__int64)&v86, (char **)v7, v10, v13, (int)v9, v12);
            goto LABEL_56;
          }
          v86 = v88;
          v87 = 0x800000000LL;
          if ( (_DWORD)v10 )
LABEL_62:
            sub_18E63F0((__int64)&v86, (char **)v7, v10, v13, (int)v9, v12);
        }
        v89 = *(unsigned __int64 **)(v7 + 144);
        v90 = *(_DWORD *)(v7 + 152);
        sub_18E63F0(v7, v68, v10, v13, (int)v9, v12);
        *(_QWORD *)(v7 + 144) = *(_QWORD *)(v7 + 304);
        *(_DWORD *)(v7 + 152) = *(_DWORD *)(v7 + 312);
        sub_18E63F0((__int64)v68, &v86, v26, v27, v28, v29);
        v30 = v89;
        v31 = v86;
        *(_QWORD *)(v7 + 304) = v89;
        v25 = v30;
        *(_DWORD *)(v7 + 312) = v90;
        if ( v31 != v88 )
          goto LABEL_18;
LABEL_20:
        v22 = *(unsigned __int64 **)(v8 - 16);
LABEL_21:
        v32 = *(_QWORD **)(v7 + 144);
        v33 = v8;
        v34 = v25;
        v69 = v8;
        v71 = v7;
        v8 = (unsigned __int64)v68;
        v35 = v22;
        v36 = *v32;
        while ( 1 )
        {
          v73 = v8;
          if ( v36 == *v34 )
          {
            v79 = v32;
            v38 = sub_16A9900((__int64)(v34 + 3), v32 + 3);
            v32 = v79;
            v37 = v38 >> 31;
          }
          else
          {
            LOBYTE(v37) = *(_DWORD *)(*v34 + 8) >> 8 < *(_DWORD *)(v36 + 8) >> 8;
          }
          if ( (_BYTE)v37 )
            goto LABEL_24;
          v39 = v33 - 160;
          v40 = (__int64)(v32 + 3);
          while ( 1 )
          {
            v33 = v39;
            if ( v36 == *v35 )
            {
              v75 = v39;
              v80 = v40;
              v43 = sub_16A9900(v40, v35 + 3);
              v40 = v80;
              v39 = v75;
              v42 = v43 >> 31;
            }
            else
            {
              v41 = *(_DWORD *)(v36 + 8) >> 8;
              LOBYTE(v42) = (unsigned int)v41 < *(_DWORD *)(*v35 + 8) >> 8;
            }
            v39 -= 160;
            if ( !(_BYTE)v42 )
              break;
            v35 = *(unsigned __int64 **)(v39 + 144);
          }
          if ( v8 >= v33 )
            break;
          v86 = v88;
          v87 = 0x800000000LL;
          if ( *(_DWORD *)(v8 + 8) )
          {
            sub_18E63F0((__int64)&v86, (char **)v8, v41, v39, v40, v21);
            v34 = *(unsigned __int64 **)(v8 + 144);
          }
          v44 = *(_DWORD *)(v8 + 152);
          v89 = v34;
          v90 = v44;
          sub_18E63F0(v8, (char **)v33, v41, v39, v40, v21);
          *(_QWORD *)(v8 + 144) = *(_QWORD *)(v33 + 144);
          *(_DWORD *)(v8 + 152) = *(_DWORD *)(v33 + 152);
          sub_18E63F0(v33, &v86, v45, v46, v47, v48);
          v49 = v86;
          *(_QWORD *)(v33 + 144) = v89;
          *(_DWORD *)(v33 + 152) = v90;
          if ( v49 != v88 )
            _libc_free((unsigned __int64)v49);
          v35 = *(unsigned __int64 **)(v33 - 16);
          v32 = *(_QWORD **)(v71 + 144);
          v36 = *v32;
LABEL_24:
          v34 = *(unsigned __int64 **)(v8 + 304);
          v8 += 160LL;
        }
        v7 = v71;
        sub_18E7670(v8, v69, v70, v39, v40, v21, v67);
        if ( (__int64)(v8 - v71) <= 2560 )
          return;
        if ( !v70 )
          goto LABEL_52;
      }
    }
    v73 = a2;
LABEL_52:
    v85 = a7;
    v53 = v73 - 160;
    LOBYTE(v86) = a7;
    sub_18E6E80(v7, v73, (__int64)&v86, a4, a5, a6);
    do
    {
      v56 = v53;
      v57 = v53;
      v58 = v53 - v7;
      v53 -= 160;
      sub_18E7070(v7, v57, v56, (__int64)&v85, v54, v55);
    }
    while ( v58 > 160 );
  }
}
