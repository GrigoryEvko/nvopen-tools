// Function: sub_1B0B080
// Address: 0x1b0b080
//
char __fastcall sub_1B0B080(__int64 a1, unsigned int a2, __int64 a3, __m128i a4, __m128i a5, __int64 a6, _QWORD *a7)
{
  __int64 v7; // r14
  unsigned int v11; // r15d
  _QWORD *v12; // rax
  unsigned __int64 v13; // rsi
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rbx
  __int64 i; // r12
  unsigned int v20; // eax
  _QWORD *v21; // r12
  int v22; // eax
  int v23; // edx
  unsigned int v24; // eax
  __int64 *v25; // rbx
  __int64 v26; // r15
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // r14d
  __int64 v32; // r15
  unsigned int v33; // eax
  unsigned int v34; // edx
  _DWORD *v35; // r8
  _DWORD *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // r9d
  __int64 v43; // r8
  __int64 v44; // rax
  unsigned int j; // r15d
  unsigned int v46; // eax
  __int64 v47; // r8
  const void *v48; // rcx
  __int64 v49; // r8
  const void *v50; // rax
  __int64 v51; // r15
  void *v52; // r10
  int v53; // r8d
  size_t v54; // r11
  __int64 v56; // [rsp+8h] [rbp-158h]
  void *v57; // [rsp+10h] [rbp-150h]
  __int64 v58; // [rsp+10h] [rbp-150h]
  int v59; // [rsp+18h] [rbp-148h]
  const void *v60; // [rsp+18h] [rbp-148h]
  const void *v61; // [rsp+20h] [rbp-140h]
  int v62; // [rsp+20h] [rbp-140h]
  __int64 v63; // [rsp+30h] [rbp-130h]
  __int64 v64; // [rsp+30h] [rbp-130h]
  __int64 *v65; // [rsp+38h] [rbp-128h]
  __int64 v66; // [rsp+48h] [rbp-118h]
  unsigned int v67; // [rsp+50h] [rbp-110h]
  unsigned int v68; // [rsp+54h] [rbp-10Ch]
  _QWORD *v69; // [rsp+58h] [rbp-108h]
  __int64 v70; // [rsp+58h] [rbp-108h]
  __int64 v71; // [rsp+58h] [rbp-108h]
  unsigned int v72; // [rsp+60h] [rbp-100h]
  unsigned int v73; // [rsp+68h] [rbp-F8h]
  __int64 *v74; // [rsp+68h] [rbp-F8h]
  bool v75; // [rsp+7Fh] [rbp-E1h] BYREF
  void *src; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v77; // [rsp+88h] [rbp-D8h]
  _BYTE v78[32]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v79; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+B8h] [rbp-A8h]
  _QWORD v81[4]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v82; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-78h]
  __int64 v84; // [rsp+F0h] [rbp-70h] BYREF
  char v85; // [rsp+130h] [rbp-30h] BYREF

  v7 = a3;
  v11 = *(_DWORD *)(a3 + 24);
  *(_DWORD *)(a3 + 24) = 0;
  LOBYTE(v12) = sub_1B0A520(a1);
  if ( (_BYTE)v12 )
  {
    v12 = *(_QWORD **)(a1 + 16);
    if ( *(_QWORD **)(a1 + 8) == v12 )
    {
      v13 = sub_16D5D50();
      v12 = *(_QWORD **)&dword_4FA0208[2];
      if ( *(_QWORD *)&dword_4FA0208[2] )
      {
        v14 = dword_4FA0208;
        do
        {
          while ( 1 )
          {
            v15 = v12[2];
            v16 = v12[3];
            if ( v13 <= v12[4] )
              break;
            v12 = (_QWORD *)v12[3];
            if ( !v16 )
              goto LABEL_9;
          }
          v14 = v12;
          v12 = (_QWORD *)v12[2];
        }
        while ( v15 );
LABEL_9:
        v12 = dword_4FA0208;
        if ( v14 != dword_4FA0208 && v13 >= *((_QWORD *)v14 + 4) )
        {
          v12 = (_QWORD *)*((_QWORD *)v14 + 7);
          v35 = v14 + 12;
          if ( v12 )
          {
            v36 = v14 + 12;
            do
            {
              while ( 1 )
              {
                v37 = v12[2];
                v38 = v12[3];
                if ( *((_DWORD *)v12 + 8) >= dword_4FB68A8 )
                  break;
                v12 = (_QWORD *)v12[3];
                if ( !v38 )
                  goto LABEL_50;
              }
              v36 = v12;
              v12 = (_QWORD *)v12[2];
            }
            while ( v37 );
LABEL_50:
            if ( v36 != v35 && dword_4FB68A8 >= v36[8] )
            {
              LODWORD(v12) = v36[9];
              if ( (int)v12 > 0 )
              {
                LOBYTE(v12) = dword_4FB6940;
                *(_DWORD *)(v7 + 24) = dword_4FB6940;
                return (char)v12;
              }
            }
          }
        }
      }
      if ( *(_BYTE *)(v7 + 50) )
      {
        LOBYTE(v12) = 2 * a2;
        if ( 2 * a2 <= *(_DWORD *)v7 )
        {
          if ( dword_4FB6A20 )
          {
            v82 = 0;
            v17 = &v84;
            v83 = 1;
            do
            {
              *v17 = -8;
              v17 += 2;
            }
            while ( v17 != (__int64 *)&v85 );
            v73 = a2;
            v69 = a7;
            v18 = sub_13FCB50(a1);
            for ( i = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 48LL); ; i = *(_QWORD *)(i + 8) )
            {
              if ( !i )
                BUG();
              if ( *(_BYTE *)(i - 8) != 77 )
                break;
              v20 = sub_1B0A950(i - 24, a1, v18, (__int64)&v82);
              if ( v20 != -1 && v11 < v20 )
                v11 = v20;
            }
            v21 = v69;
            v22 = *(_DWORD *)v7 / v73;
            v23 = dword_4FB6A20;
            v74 = *(__int64 **)(a1 + 40);
            v24 = v22 - 1;
            if ( v24 <= dword_4FB6A20 )
              v23 = v24;
            v67 = v23;
            if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 40) )
            {
              v68 = v11;
              v25 = *(__int64 **)(a1 + 32);
              v72 = 0;
              v66 = v7;
              while ( 1 )
              {
                v26 = *v25;
                v27 = sub_157EBA0(*v25);
                v28 = v27;
                if ( *(_BYTE *)(v27 + 16) != 26 )
                  goto LABEL_37;
                if ( (*(_DWORD *)(v27 + 20) & 0xFFFFFFF) == 1 )
                  goto LABEL_37;
                if ( v26 == sub_13FCB50(a1) )
                  goto LABEL_37;
                v29 = *(_QWORD *)(v28 - 72);
                if ( *(_BYTE *)(v29 + 16) != 75 )
                  goto LABEL_37;
                v30 = *(_QWORD *)(v29 - 48);
                if ( !v30 )
                  goto LABEL_37;
                v70 = *(_QWORD *)(v29 - 24);
                if ( !v70 )
                  goto LABEL_37;
                v31 = *(_WORD *)(v29 + 18) & 0x7FFF;
                v32 = sub_146F1B0((__int64)v21, v30);
                v71 = sub_146F1B0((__int64)v21, v70);
                if ( (unsigned __int8)sub_147A340((__int64)v21, v31, v32, v71) )
                  goto LABEL_37;
                v33 = sub_15FF0F0(v31);
                if ( (unsigned __int8)sub_147A340((__int64)v21, v33, v32, v71) )
                  goto LABEL_37;
                if ( *(_WORD *)(v32 + 24) != 7 )
                {
                  if ( *(_WORD *)(v71 + 24) != 7 )
                    goto LABEL_37;
                  v31 = sub_15FF5D0(v31);
                  v39 = v32;
                  v32 = v71;
                  v71 = v39;
                }
                if ( *(_QWORD *)(v32 + 40) == 2
                  && a1 == *(_QWORD *)(v32 + 48)
                  && (unsigned __int8)sub_14798E0((__int64)v21, v32, v31, &v75) )
                {
                  v40 = sub_1456040(v32);
                  v41 = sub_145CF80((__int64)v21, v40, v72, 0);
                  v65 = sub_1487810(v32, v41, v21, a4, a5);
                  if ( !(unsigned __int8)sub_147A340((__int64)v21, v31, (__int64)v65, v71) )
                    v31 = sub_15FF0F0(v31);
                  v43 = *(_QWORD *)(v32 + 40);
                  v44 = *(_QWORD *)(v32 + 32);
                  if ( v43 == 2 )
                  {
                    v63 = *(_QWORD *)(v44 + 8);
LABEL_64:
                    for ( j = v72; v67 > j && (unsigned __int8)sub_147A340((__int64)v21, v31, (__int64)v65, v71); ++j )
                    {
                      v81[0] = v65;
                      v79 = v81;
                      v81[1] = v63;
                      v80 = 0x200000002LL;
                      v65 = sub_147DD40((__int64)v21, (__int64 *)&v79, 0, 0, a4, a5);
                      if ( v79 != v81 )
                        _libc_free((unsigned __int64)v79);
                    }
                    if ( v72 < j )
                    {
                      v46 = sub_15FF0F0(v31);
                      if ( !(unsigned __int8)sub_147A340((__int64)v21, v46, (__int64)v65, v71) )
                        j = v72;
                      v72 = j;
                    }
                    goto LABEL_37;
                  }
                  v47 = 8 * v43;
                  v64 = *(_QWORD *)(v32 + 48);
                  v48 = (const void *)(v44 + v47);
                  v49 = v47 - 8;
                  v50 = (const void *)(v44 + 8);
                  v61 = v48;
                  v51 = v49 >> 3;
                  src = v78;
                  v77 = 0x300000000LL;
                  if ( (unsigned __int64)v49 > 0x18 )
                  {
                    v58 = v49;
                    v60 = v50;
                    sub_16CD150((__int64)&src, v78, v49 >> 3, 8, v49, v42);
                    v49 = v58;
                    v50 = v60;
                  }
                  if ( v61 != v50 )
                    memcpy((char *)src + 8 * (unsigned int)v77, v50, v49);
                  v52 = src;
                  v80 = 0x400000000LL;
                  LODWORD(v77) = v51 + v77;
                  v53 = v77;
                  v79 = v81;
                  v54 = 8LL * (unsigned int)v77;
                  if ( (unsigned int)v77 > 4uLL )
                  {
                    v56 = 8LL * (unsigned int)v77;
                    v57 = src;
                    v59 = v77;
                    sub_16CD150((__int64)&v79, v81, (unsigned int)v77, 8, v77, (int)&v79);
                    v53 = v59;
                    v52 = v57;
                    v54 = v56;
                  }
                  else if ( !v54 )
                  {
LABEL_80:
                    LODWORD(v80) = v53 + v80;
                    v63 = sub_14785F0((__int64)v21, &v79, v64, 0);
                    if ( v79 != v81 )
                      _libc_free((unsigned __int64)v79);
                    if ( src != v78 )
                      _libc_free((unsigned __int64)src);
                    goto LABEL_64;
                  }
                  v62 = v53;
                  memcpy(&v79[(unsigned int)v80], v52, v54);
                  v53 = v62;
                  goto LABEL_80;
                }
LABEL_37:
                if ( v74 == ++v25 )
                {
                  v11 = v68;
                  v7 = v66;
                  if ( v68 < v72 )
                    v11 = v72;
                  break;
                }
              }
            }
            LOBYTE(v12) = v83 & 1;
            if ( !v11 )
            {
              if ( (_BYTE)v12 )
                return (char)v12;
              goto LABEL_44;
            }
            v34 = v67;
            if ( v11 <= v67 )
              v34 = v11;
            *(_DWORD *)(v7 + 24) = v34;
            if ( !(_BYTE)v12 )
LABEL_44:
              LOBYTE(v12) = j___libc_free_0(v84);
          }
        }
      }
    }
  }
  return (char)v12;
}
