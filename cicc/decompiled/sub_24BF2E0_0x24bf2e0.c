// Function: sub_24BF2E0
// Address: 0x24bf2e0
//
void __fastcall sub_24BF2E0(__int64 a1, __int64 a2, const char *a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // r14
  __int64 v7; // r13
  _QWORD *v9; // r10
  __int64 v10; // r9
  unsigned int *v11; // rcx
  __int64 v12; // rdx
  __int64 *v13; // r13
  __int64 v14; // r8
  _QWORD *v15; // r15
  __int64 *v16; // rbx
  __int64 v17; // r12
  unsigned int *v18; // rsi
  __int64 *v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // r14
  size_t v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r13
  unsigned int *v33; // rax
  int v34; // ecx
  unsigned int *v35; // rdx
  __int64 v36; // r8
  __int64 v37; // rsi
  int v38; // edi
  __int64 v39; // rax
  _QWORD *v40; // r15
  __int64 v41; // r12
  int v42; // edx
  unsigned int v43; // ecx
  unsigned __int8 v44; // al
  int v45; // r12d
  unsigned __int64 v46; // r13
  unsigned int *v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 *v50; // rax
  __int64 *v51; // rax
  unsigned __int64 v52; // rsi
  __int64 v53; // [rsp+0h] [rbp-160h]
  unsigned int v54; // [rsp+8h] [rbp-158h]
  __int64 *v55; // [rsp+10h] [rbp-150h]
  __int64 v56; // [rsp+10h] [rbp-150h]
  char *s; // [rsp+20h] [rbp-140h]
  __int64 v58; // [rsp+28h] [rbp-138h]
  unsigned __int64 v59; // [rsp+30h] [rbp-130h]
  __int64 v60; // [rsp+30h] [rbp-130h]
  __int64 v61; // [rsp+30h] [rbp-130h]
  char v63[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v64; // [rsp+60h] [rbp-100h]
  _QWORD v65[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v66; // [rsp+90h] [rbp-D0h]
  unsigned int *v67; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-B8h]
  _BYTE v69[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+D0h] [rbp-90h]
  __int64 v71; // [rsp+D8h] [rbp-88h]
  __int64 v72; // [rsp+E0h] [rbp-80h]
  __int64 v73; // [rsp+E8h] [rbp-78h]
  void **v74; // [rsp+F0h] [rbp-70h]
  void **v75; // [rsp+F8h] [rbp-68h]
  __int64 v76; // [rsp+100h] [rbp-60h]
  int v77; // [rsp+108h] [rbp-58h]
  __int16 v78; // [rsp+10Ch] [rbp-54h]
  char v79; // [rsp+10Eh] [rbp-52h]
  __int64 v80; // [rsp+110h] [rbp-50h]
  __int64 v81; // [rsp+118h] [rbp-48h]
  void *v82; // [rsp+120h] [rbp-40h] BYREF
  void *v83; // [rsp+128h] [rbp-38h] BYREF

  v6 = a1;
  v7 = a2;
  v67 = (unsigned int *)v69;
  v9 = (_QWORD *)sub_B2BE50(a1);
  v68 = 0x600000000LL;
  v10 = (__int64)&a4[a5];
  if ( (__int64 *)v10 == a4 )
  {
    v12 = 0;
    v18 = (unsigned int *)v69;
  }
  else
  {
    v11 = (unsigned int *)v69;
    v12 = 0;
    v13 = &a4[a5];
    v14 = *(_QWORD *)(*a4 + 8);
    s = (char *)a3;
    v15 = v9;
    v55 = a4;
    v16 = a4 + 1;
    v17 = v14;
    while ( 1 )
    {
      *(_QWORD *)&v11[2 * v12] = v17;
      v12 = (unsigned int)(v68 + 1);
      LODWORD(v68) = v68 + 1;
      if ( v13 == v16 )
        break;
      v17 = *(_QWORD *)(*v16 + 8);
      if ( v12 + 1 > (unsigned __int64)HIDWORD(v68) )
      {
        sub_C8D5F0((__int64)&v67, v69, v12 + 1, 8u, v14, v10);
        v12 = (unsigned int)v68;
      }
      v11 = v67;
      ++v16;
    }
    v9 = v15;
    v7 = a2;
    a3 = s;
    a4 = v55;
    v6 = a1;
    v18 = v67;
  }
  v58 = v12;
  v19 = (__int64 *)sub_BCB120(v9);
  v20 = sub_BCF480(v19, v18, v58, 0);
  v21 = v20;
  if ( v67 != (unsigned int *)v69 )
  {
    v59 = v20;
    _libc_free((unsigned __int64)v67);
    v21 = v59;
  }
  v22 = *(_QWORD *)(v6 + 40);
  v60 = v21;
  v23 = strlen(a3);
  v24 = sub_BA8CA0(v22, (__int64)a3, v23, v60);
  v61 = v25;
  v26 = v24;
  v27 = sub_BD5C60(v7);
  v79 = 7;
  v73 = v27;
  v74 = &v82;
  v75 = &v83;
  v78 = 512;
  v68 = 0x200000000LL;
  v67 = (unsigned int *)v69;
  v82 = &unk_49DA100;
  v76 = 0;
  v77 = 0;
  v83 = &unk_49DA0B0;
  v28 = *(_QWORD *)(v7 + 40);
  v80 = 0;
  v70 = v28;
  v81 = 0;
  v71 = v7 + 24;
  LOWORD(v72) = 0;
  v29 = *(_QWORD *)sub_B46C60(v7);
  v65[0] = v29;
  if ( v29 && (sub_B96E90((__int64)v65, v29, 1), (v32 = v65[0]) != 0) )
  {
    v33 = v67;
    v34 = v68;
    v35 = &v67[4 * (unsigned int)v68];
    if ( v67 != v35 )
    {
      while ( 1 )
      {
        v31 = *v33;
        if ( !(_DWORD)v31 )
          break;
        v33 += 4;
        if ( v35 == v33 )
          goto LABEL_53;
      }
      *((_QWORD *)v33 + 1) = v65[0];
      goto LABEL_17;
    }
LABEL_53:
    if ( (unsigned int)v68 >= (unsigned __int64)HIDWORD(v68) )
    {
      v52 = (unsigned int)v68 + 1LL;
      if ( HIDWORD(v68) < v52 )
      {
        sub_C8D5F0((__int64)&v67, v69, v52, 0x10u, v30, v31);
        v35 = &v67[4 * (unsigned int)v68];
      }
      *(_QWORD *)v35 = 0;
      *((_QWORD *)v35 + 1) = v32;
      v32 = v65[0];
      LODWORD(v68) = v68 + 1;
    }
    else
    {
      if ( v35 )
      {
        *v35 = 0;
        *((_QWORD *)v35 + 1) = v32;
        v34 = v68;
        v32 = v65[0];
      }
      LODWORD(v68) = v34 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v67, 0);
    v32 = v65[0];
  }
  if ( v32 )
LABEL_17:
    sub_B91220((__int64)v65, v32);
  v66 = 257;
  v64 = 257;
  v36 = v80 + 56 * v81;
  if ( v80 == v36 )
  {
    v38 = 0;
  }
  else
  {
    v37 = v80;
    v38 = 0;
    do
    {
      v39 = *(_QWORD *)(v37 + 40) - *(_QWORD *)(v37 + 32);
      v37 += 56;
      v38 += v39 >> 3;
    }
    while ( v36 != v37 );
  }
  v53 = v80;
  v56 = v81;
  LOBYTE(v32) = 16 * (_DWORD)v81 != 0;
  v54 = v38 + a5 + 1;
  v40 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v81) << 32) | v54);
  if ( v40 )
  {
    sub_B44260((__int64)v40, **(_QWORD **)(v26 + 16), 56, ((_DWORD)v32 << 28) | v54 & 0x7FFFFFF, 0, 0);
    v40[9] = 0;
    sub_B4A290((__int64)v40, v26, v61, a4, a5, (__int64)v65, v53, v56);
  }
  if ( (_BYTE)v78 )
  {
    v51 = (__int64 *)sub_BD5C60((__int64)v40);
    v40[9] = sub_A7A090(v40 + 9, v51, -1, 72);
  }
  if ( *(_BYTE *)v40 > 0x1Cu )
  {
    switch ( *(_BYTE *)v40 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_30;
      case 'T':
      case 'U':
      case 'V':
        v41 = v40[1];
        v42 = *(unsigned __int8 *)(v41 + 8);
        v43 = v42 - 17;
        v44 = *(_BYTE *)(v41 + 8);
        if ( (unsigned int)(v42 - 17) <= 1 )
          v44 = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
        if ( v44 <= 3u || v44 == 5 || (v44 & 0xFD) == 4 )
          goto LABEL_30;
        if ( (_BYTE)v42 == 15 )
        {
          if ( (*(_BYTE *)(v41 + 9) & 4) == 0 || !sub_BCB420(v40[1]) )
            break;
          v50 = *(__int64 **)(v41 + 16);
          v41 = *v50;
          v42 = *(unsigned __int8 *)(*v50 + 8);
          v43 = v42 - 17;
        }
        else if ( (_BYTE)v42 == 16 )
        {
          do
          {
            v41 = *(_QWORD *)(v41 + 24);
            LOBYTE(v42) = *(_BYTE *)(v41 + 8);
          }
          while ( (_BYTE)v42 == 16 );
          v43 = (unsigned __int8)v42 - 17;
        }
        if ( v43 <= 1 )
          LOBYTE(v42) = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
        if ( (unsigned __int8)v42 <= 3u || (_BYTE)v42 == 5 || (v42 & 0xFD) == 4 )
        {
LABEL_30:
          v45 = v77;
          if ( v76 )
            sub_B99FD0((__int64)v40, 3u, v76);
          sub_B45150((__int64)v40, v45);
        }
        break;
      default:
        break;
    }
  }
  (*((void (__fastcall **)(void **, _QWORD *, char *, __int64, __int64))*v75 + 2))(v75, v40, v63, v71, v72);
  v46 = (unsigned __int64)v67;
  v47 = &v67[4 * (unsigned int)v68];
  if ( v67 != v47 )
  {
    do
    {
      v48 = *(_QWORD *)(v46 + 8);
      v49 = *(_DWORD *)v46;
      v46 += 16LL;
      sub_B99FD0((__int64)v40, v49, v48);
    }
    while ( v47 != (unsigned int *)v46 );
  }
  nullsub_61();
  v82 = &unk_49DA100;
  nullsub_63();
  if ( v67 != (unsigned int *)v69 )
    _libc_free((unsigned __int64)v67);
}
