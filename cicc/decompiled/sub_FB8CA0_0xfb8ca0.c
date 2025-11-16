// Function: sub_FB8CA0
// Address: 0xfb8ca0
//
__int64 __fastcall sub_FB8CA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned int a7)
{
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  int v11; // r13d
  int v12; // r15d
  unsigned int i; // r13d
  __int64 *v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 v23; // rax
  int v24; // ecx
  unsigned int *v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // r9
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rbx
  unsigned int *v31; // rbx
  unsigned int *v32; // r12
  __int64 v33; // rdx
  __int64 v34; // r9
  _BYTE *v35; // r8
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // rbx
  __int64 *v39; // r15
  __int64 v40; // r12
  __int64 *v41; // rax
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // r9
  unsigned __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 *v48; // rbx
  __int64 *v49; // r12
  __int64 *v50; // rdx
  __int64 v51; // r13
  _QWORD *v52; // rdi
  unsigned __int64 v53; // rsi
  __int64 v54; // [rsp+8h] [rbp-198h]
  __int64 v55; // [rsp+8h] [rbp-198h]
  __int64 v58; // [rsp+30h] [rbp-170h]
  __int64 v59; // [rsp+30h] [rbp-170h]
  __int64 v60; // [rsp+30h] [rbp-170h]
  __int64 v64; // [rsp+60h] [rbp-140h]
  __int64 v65; // [rsp+60h] [rbp-140h]
  _BYTE *v66; // [rsp+60h] [rbp-140h]
  _BYTE *v67; // [rsp+60h] [rbp-140h]
  _BYTE *v68; // [rsp+70h] [rbp-130h] BYREF
  __int64 v69; // [rsp+78h] [rbp-128h]
  _BYTE v70[16]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v71; // [rsp+90h] [rbp-110h]
  __int64 v72; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v73; // [rsp+A8h] [rbp-F8h]
  __int64 v74; // [rsp+B0h] [rbp-F0h]
  __int64 v75; // [rsp+B8h] [rbp-E8h]
  __int64 *v76; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v77; // [rsp+C8h] [rbp-D8h]
  _BYTE v78[16]; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned int *v79; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+E8h] [rbp-B8h]
  _BYTE v81[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v82; // [rsp+110h] [rbp-90h]
  __int64 v83; // [rsp+118h] [rbp-88h]
  __int64 v84; // [rsp+120h] [rbp-80h]
  __int64 v85; // [rsp+128h] [rbp-78h]
  void **v86; // [rsp+130h] [rbp-70h]
  void **v87; // [rsp+138h] [rbp-68h]
  __int64 v88; // [rsp+140h] [rbp-60h]
  int v89; // [rsp+148h] [rbp-58h]
  __int16 v90; // [rsp+14Ch] [rbp-54h]
  char v91; // [rsp+14Eh] [rbp-52h]
  __int64 v92; // [rsp+150h] [rbp-50h]
  __int64 v93; // [rsp+158h] [rbp-48h]
  void *v94; // [rsp+160h] [rbp-40h] BYREF
  void *v95; // [rsp+168h] [rbp-38h] BYREF

  v7 = 0;
  v8 = a2;
  v9 = a4;
  if ( a4 != a5 )
    v7 = a5;
  v10 = *(_QWORD *)(a2 + 40);
  v64 = v7;
  v76 = (__int64 *)v78;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v77 = 0x200000000LL;
  v11 = sub_B46E30(a2);
  if ( v11 )
  {
    v12 = v11;
    for ( i = 0; v12 != i; ++i )
    {
      while ( 1 )
      {
        v18 = sub_B46EC0(v8, i);
        v68 = (_BYTE *)v18;
        if ( v18 != v9 )
          break;
        v9 = 0;
LABEL_9:
        if ( v12 == ++i )
          goto LABEL_13;
      }
      if ( v18 != v64 )
      {
        sub_AA5980(v18, v10, 1u);
        if ( v68 != (_BYTE *)a4 && v68 != (_BYTE *)a5 )
        {
          if ( (_DWORD)v74 )
          {
            sub_D6CB10((__int64)&v79, (__int64)&v72, (__int64 *)&v68);
            if ( v81[16] )
            {
              v44 = (unsigned int)v77;
              v45 = (__int64)v68;
              v46 = (unsigned int)v77 + 1LL;
              if ( v46 > HIDWORD(v77) )
              {
                v60 = (__int64)v68;
                sub_C8D5F0((__int64)&v76, v78, v46, 8u, v43, (__int64)v68);
                v44 = (unsigned int)v77;
                v45 = v60;
              }
              v76[v44] = v45;
              LODWORD(v77) = v77 + 1;
            }
          }
          else
          {
            v14 = &v76[(unsigned int)v77];
            if ( v14 == sub_F8ED40(v76, (__int64)v14, (__int64 *)&v68) )
            {
              if ( v17 + 1 > (unsigned __int64)HIDWORD(v77) )
              {
                v59 = v16;
                sub_C8D5F0((__int64)&v76, v78, v17 + 1, 8u, v15, v16);
                v16 = v59;
                v14 = &v76[(unsigned int)v77];
              }
              *v14 = v16;
              v47 = (unsigned int)(v77 + 1);
              LODWORD(v77) = v47;
              if ( (unsigned int)v47 > 2 )
              {
                v58 = v9;
                v55 = v8;
                v48 = &v76[v47];
                v49 = v76;
                do
                {
                  v50 = v49++;
                  sub_D6CB10((__int64)&v79, (__int64)&v72, v50);
                }
                while ( v48 != v49 );
                v9 = v58;
                v8 = v55;
              }
            }
          }
        }
        goto LABEL_9;
      }
      v64 = 0;
    }
  }
LABEL_13:
  v85 = sub_BD5C60(v8);
  v86 = &v94;
  v87 = &v95;
  v80 = 0x200000000LL;
  v94 = &unk_49DA100;
  v79 = (unsigned int *)v81;
  v88 = 0;
  v89 = 0;
  v90 = 512;
  v91 = 7;
  v92 = 0;
  v93 = 0;
  v82 = 0;
  v83 = 0;
  LOWORD(v84) = 0;
  v95 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v79, v8);
  v19 = *(_QWORD *)(v8 + 48);
  v68 = (_BYTE *)v19;
  if ( v19 && (sub_B96E90((__int64)&v68, v19, 1), (v22 = (__int64)v68) != 0) )
  {
    v23 = (__int64)v79;
    v24 = v80;
    v25 = &v79[4 * (unsigned int)v80];
    if ( v79 != v25 )
    {
      while ( *(_DWORD *)v23 )
      {
        v23 += 16;
        if ( v25 == (unsigned int *)v23 )
          goto LABEL_53;
      }
      *(_QWORD *)(v23 + 8) = v68;
      goto LABEL_20;
    }
LABEL_53:
    if ( (unsigned int)v80 >= (unsigned __int64)HIDWORD(v80) )
    {
      v53 = (unsigned int)v80 + 1LL;
      if ( HIDWORD(v80) < v53 )
      {
        sub_C8D5F0((__int64)&v79, v81, v53, 0x10u, v20, v21);
        v25 = &v79[4 * (unsigned int)v80];
      }
      *(_QWORD *)v25 = 0;
      *((_QWORD *)v25 + 1) = v22;
      v22 = (__int64)v68;
      LODWORD(v80) = v80 + 1;
    }
    else
    {
      if ( v25 )
      {
        *v25 = 0;
        *((_QWORD *)v25 + 1) = v22;
        v24 = v80;
        v22 = (__int64)v68;
      }
      LODWORD(v80) = v24 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v79, 0);
    v22 = (__int64)v68;
  }
  if ( v22 )
LABEL_20:
    sub_B91220((__int64)&v68, v22);
  v54 = v9 | v64;
  if ( !(v9 | v64) )
  {
    if ( a4 != a5 )
    {
      v71 = 257;
      v26 = sub_BD2C40(72, 3u);
      v28 = (__int64)v26;
      if ( v26 )
        sub_B4C9A0((__int64)v26, a4, a5, a3, 3u, v27, 0, 0);
      v29 = v28;
      (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v87 + 2))(v87, v28, &v68, v83, v84);
      v30 = 4LL * (unsigned int)v80;
      if ( v79 != &v79[v30] )
      {
        v65 = v8;
        v31 = &v79[v30];
        v32 = v79;
        do
        {
          v33 = *((_QWORD *)v32 + 1);
          v29 = *v32;
          v32 += 4;
          sub_B99FD0(v28, v29, v33);
        }
        while ( v31 != v32 );
        v8 = v65;
      }
      if ( a6 != a7 )
      {
        if ( a7 | a6 )
        {
          v68 = (_BYTE *)sub_AA48A0(*(_QWORD *)(v28 + 40));
          v54 = sub_B8C2F0(&v68, a6, a7, 0);
        }
        v29 = 2;
        sub_B99FD0(v28, 2u, v54);
      }
      goto LABEL_30;
    }
LABEL_61:
    v29 = a4;
    sub_F902B0((__int64 *)&v79, a4);
    goto LABEL_30;
  }
  if ( !v9 )
    goto LABEL_61;
  if ( v64 || a4 == a5 )
  {
    v51 = sub_BD5C60(v8);
    v29 = unk_3F148B8;
    v52 = sub_BD2C40(72, unk_3F148B8);
    if ( v52 )
    {
      v29 = v51;
      sub_B4C8A0((__int64)v52, v51, v8 + 24, 0);
    }
  }
  else
  {
    v29 = a5;
    sub_F902B0((__int64 *)&v79, a5);
  }
LABEL_30:
  sub_F91380((char *)v8);
  if ( *(_QWORD *)(a1 + 8) )
  {
    v34 = (unsigned int)v77;
    v35 = v70;
    v36 = 0;
    v68 = v70;
    v69 = 0x200000000LL;
    v37 = 0;
    if ( (unsigned int)v77 > 2 )
    {
      sub_C8D5F0((__int64)&v68, v70, (unsigned int)v77, 0x10u, (__int64)v70, (unsigned int)v77);
      v36 = (unsigned int)v69;
      v34 = (unsigned int)v77;
      v35 = v70;
      v37 = (unsigned int)v69;
    }
    v38 = v76;
    v39 = &v76[v34];
    if ( v39 != v76 )
    {
      do
      {
        v40 = *v38 | 4;
        if ( v37 + 1 > (unsigned __int64)HIDWORD(v69) )
        {
          v67 = v35;
          sub_C8D5F0((__int64)&v68, v35, v37 + 1, 0x10u, (__int64)v35, v34);
          v37 = (unsigned int)v69;
          v35 = v67;
        }
        v41 = (__int64 *)&v68[16 * v37];
        ++v38;
        *v41 = v10;
        v41[1] = v40;
        v37 = (unsigned int)(v69 + 1);
        LODWORD(v69) = v69 + 1;
      }
      while ( v39 != v38 );
      v36 = (unsigned int)v37;
    }
    v29 = (__int64)v68;
    v66 = v35;
    sub_FFB3D0(*(_QWORD *)(a1 + 8), v68, v36);
    if ( v68 != v66 )
      _libc_free(v68, v29);
  }
  nullsub_61();
  v94 = &unk_49DA100;
  nullsub_63();
  if ( v79 != (unsigned int *)v81 )
    _libc_free(v79, v29);
  if ( v76 != (__int64 *)v78 )
    _libc_free(v76, v29);
  sub_C7D6A0(v73, 8LL * (unsigned int)v75, 8);
  return 1;
}
