// Function: sub_2AFA890
// Address: 0x2afa890
//
void __fastcall sub_2AFA890(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 *a5, int a6)
{
  _QWORD *v7; // rdi
  __int64 v8; // r12
  __int64 *v9; // rax
  __int64 **v10; // rdi
  __int64 v11; // r13
  __int64 *v12; // rdx
  char v13; // al
  __int64 v14; // rdi
  unsigned int *v15; // r10
  __int64 v16; // r11
  __int64 v17; // r14
  __int64 (__fastcall *v18)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v19; // rax
  _QWORD *v20; // r12
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int8 *v25; // r14
  __int64 (__fastcall *v26)(__int64, _BYTE *, _QWORD *, unsigned __int8 *); // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // r15
  __int64 v31; // r14
  __int64 v32; // r12
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rax
  int v46; // esi
  char v47; // dl
  int v48; // edx
  __int64 v49; // r12
  char v50; // al
  unsigned __int64 v51; // r13
  __int64 v52; // rax
  unsigned int v53; // r14d
  __int64 **v54; // rax
  __int64 v55; // rax
  __int64 v56; // r10
  __int64 v57; // rax
  int v58; // ecx
  char v59; // dl
  int v60; // edx
  unsigned __int64 v61; // rax
  char v62; // dl
  char v63; // r14
  _QWORD *v64; // rax
  __int64 v65; // r9
  __int64 v66; // r12
  __int64 v67; // r13
  __int64 v68; // rbx
  __int64 v69; // r13
  __int64 v70; // rdx
  unsigned int v71; // esi
  int v72; // r8d
  int v73; // edi
  __int64 v74; // [rsp+0h] [rbp-100h]
  __int64 v77; // [rsp+18h] [rbp-E8h]
  __int64 v79; // [rsp+28h] [rbp-D8h]
  __int64 v80; // [rsp+30h] [rbp-D0h]
  unsigned int *v81; // [rsp+38h] [rbp-C8h]
  unsigned int *v82; // [rsp+38h] [rbp-C8h]
  __int64 v83; // [rsp+40h] [rbp-C0h]
  unsigned int *v84; // [rsp+40h] [rbp-C0h]
  __int64 v85; // [rsp+40h] [rbp-C0h]
  unsigned int v87; // [rsp+64h] [rbp-9Ch]
  unsigned __int64 v88; // [rsp+68h] [rbp-98h]
  _QWORD v89[4]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v90; // [rsp+90h] [rbp-70h]
  _QWORD v91[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v92; // [rsp+C0h] [rbp-40h]

  v7 = *(_QWORD **)(a2 + 72);
  if ( a6 == 1 )
  {
    v8 = a1[21];
    v9 = (__int64 *)sub_BCB2C0(v7);
  }
  else
  {
    v8 = a1[20];
    v9 = (__int64 *)sub_BCB2B0(v7);
  }
  v10 = (__int64 **)sub_BCDA70(v9, v8);
  v88 = sub_ACA8A0(v10);
  v11 = *a5;
  v79 = a5[1];
  if ( *a5 != v79 )
  {
    v87 = 0;
    while ( 1 )
    {
      v13 = *((_BYTE *)a4 + 32);
      if ( v13 )
      {
        if ( v13 == 1 )
        {
          v89[0] = ".extract";
          v90 = 259;
        }
        else
        {
          if ( *((_BYTE *)a4 + 33) == 1 )
          {
            v12 = (__int64 *)*a4;
            v80 = a4[1];
          }
          else
          {
            v12 = a4;
            v13 = 2;
          }
          v89[0] = v12;
          LOBYTE(v90) = v13;
          v89[1] = v80;
          v89[2] = ".extract";
          HIBYTE(v90) = 3;
        }
      }
      else
      {
        v90 = 256;
      }
      v14 = *(_QWORD *)(a2 + 80);
      v15 = *(unsigned int **)v11;
      v16 = *(unsigned int *)(v11 + 8);
      v17 = *a3;
      v18 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v14 + 80LL);
      if ( v18 != sub_92FAE0 )
        break;
      if ( *(_BYTE *)v17 <= 0x15u )
      {
        v81 = *(unsigned int **)v11;
        v83 = *(unsigned int *)(v11 + 8);
        v19 = sub_AAADB0(*a3, *(unsigned int **)v11, v83);
        v16 = v83;
        v15 = v81;
        v20 = (_QWORD *)v19;
        goto LABEL_12;
      }
LABEL_31:
      v77 = v16;
      v92 = 257;
      v84 = v15;
      v20 = sub_BD2C40(104, 1u);
      if ( v20 )
      {
        v35 = sub_B501B0(*(_QWORD *)(v17 + 8), v84, v77);
        sub_B44260((__int64)v20, v35, 64, 1u, 0, 0);
        if ( *(v20 - 4) )
        {
          v36 = *(v20 - 3);
          *(_QWORD *)*(v20 - 2) = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(v20 - 2);
        }
        *(v20 - 4) = v17;
        v37 = *(_QWORD *)(v17 + 16);
        *(v20 - 3) = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = v20 - 3;
        *(v20 - 2) = v17 + 16;
        *(_QWORD *)(v17 + 16) = v20 - 4;
        v20[9] = v20 + 11;
        v20[10] = 0x400000000LL;
        sub_B50030((__int64)v20, v84, v77, (__int64)v91);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v20,
        v89,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v38 = *(_QWORD *)a2;
      v39 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v39 )
      {
        do
        {
          v40 = *(_QWORD *)(v38 + 8);
          v41 = *(_DWORD *)v38;
          v38 += 16;
          sub_B99FD0((__int64)v20, v41, v40);
        }
        while ( v39 != v38 );
      }
LABEL_13:
      v21 = *(_QWORD **)(a2 + 72);
      v90 = 257;
      v22 = sub_BCB2D0(v21);
      v23 = sub_ACD640(v22, v87, 0);
      v24 = *(_QWORD *)(a2 + 80);
      v25 = (unsigned __int8 *)v23;
      v26 = *(__int64 (__fastcall **)(__int64, _BYTE *, _QWORD *, unsigned __int8 *))(*(_QWORD *)v24 + 104LL);
      if ( (char *)v26 == (char *)sub_948040 )
      {
        v27 = 0;
        if ( *(_BYTE *)v88 <= 0x15u )
          v27 = v88;
        if ( *(_BYTE *)v20 > 0x15u || *v25 > 0x15u || !v27 )
        {
LABEL_26:
          v92 = 257;
          v29 = sub_BD2C40(72, 3u);
          v30 = (__int64)v29;
          if ( v29 )
            sub_B4DFA0((__int64)v29, v88, (__int64)v20, (__int64)v25, (__int64)v91, 0, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v30,
            v89,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64));
          v31 = *(_QWORD *)a2;
          v32 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v32 )
          {
            do
            {
              v33 = *(_QWORD *)(v31 + 8);
              v34 = *(_DWORD *)v31;
              v31 += 16;
              sub_B99FD0(v30, v34, v33);
            }
            while ( v32 != v31 );
          }
          v88 = v30;
          goto LABEL_22;
        }
        v28 = sub_AD5A90(v27, v20, v25, 0);
      }
      else
      {
        v28 = v26(v24, (_BYTE *)v88, v20, v25);
      }
      if ( !v28 )
        goto LABEL_26;
      v88 = v28;
LABEL_22:
      ++v87;
      v11 += 88;
      if ( v79 == v11 )
        goto LABEL_43;
    }
    v82 = *(unsigned int **)v11;
    v85 = *(unsigned int *)(v11 + 8);
    v42 = v18(v14, (_BYTE *)v17, (__int64)v15, v16);
    v15 = v82;
    v16 = v85;
    v20 = (_QWORD *)v42;
LABEL_12:
    if ( v20 )
      goto LABEL_13;
    goto LABEL_31;
  }
LABEL_43:
  v43 = *(_QWORD **)(a2 + 72);
  v92 = 257;
  v44 = sub_BCB2D0(v43);
  v45 = *(_QWORD *)(v88 + 8);
  if ( v44 != v45 )
  {
    v46 = *(unsigned __int8 *)(v45 + 8);
    v47 = *(_BYTE *)(v45 + 8);
    if ( (unsigned int)(v46 - 17) > 1 )
    {
      if ( (_BYTE)v46 != 14 )
        goto LABEL_80;
    }
    else if ( *(_BYTE *)(**(_QWORD **)(v45 + 16) + 8LL) != 14 )
    {
      goto LABEL_46;
    }
    v72 = *(unsigned __int8 *)(v44 + 8);
    if ( (unsigned int)(v72 - 17) <= 1 )
      LOBYTE(v72) = *(_BYTE *)(**(_QWORD **)(v44 + 16) + 8LL);
    if ( (_BYTE)v72 == 12 )
    {
      v88 = sub_2AF9000((__int64 *)a2, 0x2Fu, v88, (__int64 **)v44, (__int64)v91, 0, v89[0], 0);
      goto LABEL_53;
    }
LABEL_46:
    if ( v46 == 18 )
    {
LABEL_47:
      v47 = *(_BYTE *)(**(_QWORD **)(v45 + 16) + 8LL);
LABEL_48:
      if ( v47 != 12 )
        goto LABEL_52;
      v48 = *(unsigned __int8 *)(v44 + 8);
      if ( (unsigned int)(v48 - 17) <= 1 )
        LOBYTE(v48) = *(_BYTE *)(**(_QWORD **)(v44 + 16) + 8LL);
      if ( (_BYTE)v48 == 14 )
        v88 = sub_2AF9000((__int64 *)a2, 0x30u, v88, (__int64 **)v44, (__int64)v91, 0, v89[0], 0);
      else
LABEL_52:
        v88 = sub_2AF9000((__int64 *)a2, 0x31u, v88, (__int64 **)v44, (__int64)v91, 0, v89[0], 0);
      goto LABEL_53;
    }
LABEL_80:
    if ( v46 != 17 )
      goto LABEL_48;
    goto LABEL_47;
  }
LABEL_53:
  v49 = *a5;
  v50 = *((_BYTE *)a4 + 32);
  if ( v50 )
  {
    if ( v50 == 1 )
    {
      v91[0] = ".gep";
      v92 = 259;
    }
    else
    {
      if ( *((_BYTE *)a4 + 33) == 1 )
      {
        v74 = a4[1];
        a4 = (__int64 *)*a4;
      }
      else
      {
        v50 = 2;
      }
      LOBYTE(v92) = v50;
      HIBYTE(v92) = 3;
      v91[0] = a4;
      v91[1] = v74;
      v91[2] = ".gep";
    }
  }
  else
  {
    v92 = 256;
  }
  v51 = sub_921130(
          (unsigned int **)a2,
          a1[12],
          a1[11],
          *(_BYTE ***)(v49 + 32),
          *(unsigned int *)(v49 + 40),
          (__int64)v91,
          3u);
  v52 = *(_QWORD *)(a1[11] + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 <= 1 )
    v52 = **(_QWORD **)(v52 + 16);
  v53 = *(_DWORD *)(v52 + 8);
  v54 = (__int64 **)sub_BCB2D0(*(_QWORD **)(a2 + 72));
  v55 = sub_BCE760(v54, v53 >> 8);
  v92 = 257;
  v56 = v55;
  v57 = *(_QWORD *)(v51 + 8);
  if ( v56 != v57 )
  {
    v58 = *(unsigned __int8 *)(v57 + 8);
    v59 = *(_BYTE *)(v57 + 8);
    if ( (unsigned int)(v58 - 17) > 1 )
    {
      if ( (_BYTE)v58 != 14 )
        goto LABEL_83;
    }
    else if ( *(_BYTE *)(**(_QWORD **)(v57 + 16) + 8LL) != 14 )
    {
      goto LABEL_64;
    }
    v73 = *(unsigned __int8 *)(v56 + 8);
    if ( (unsigned int)(v73 - 17) <= 1 )
      LOBYTE(v73) = *(_BYTE *)(**(_QWORD **)(v56 + 16) + 8LL);
    if ( (_BYTE)v73 == 12 )
    {
      v51 = sub_2AF9000((__int64 *)a2, 0x2Fu, v51, (__int64 **)v56, (__int64)v91, 0, v89[0], 0);
      goto LABEL_71;
    }
LABEL_64:
    if ( v58 == 18 )
    {
LABEL_65:
      v59 = *(_BYTE *)(**(_QWORD **)(v57 + 16) + 8LL);
LABEL_66:
      if ( v59 != 12 )
        goto LABEL_70;
      v60 = *(unsigned __int8 *)(v56 + 8);
      if ( (unsigned int)(v60 - 17) <= 1 )
        LOBYTE(v60) = *(_BYTE *)(**(_QWORD **)(v56 + 16) + 8LL);
      if ( (_BYTE)v60 == 14 )
        v51 = sub_2AF9000((__int64 *)a2, 0x30u, v51, (__int64 **)v56, (__int64)v91, 0, v89[0], 0);
      else
LABEL_70:
        v51 = sub_2AF9000((__int64 *)a2, 0x31u, v51, (__int64 **)v56, (__int64)v91, 0, v89[0], 0);
      goto LABEL_71;
    }
LABEL_83:
    if ( v58 != 17 )
      goto LABEL_66;
    goto LABEL_65;
  }
LABEL_71:
  v61 = *(_QWORD *)(v49 + 80);
  v62 = -1;
  if ( v61 )
  {
    _BitScanReverse64(&v61, v61);
    v62 = 63 - (v61 ^ 0x3F);
  }
  v92 = 257;
  v63 = v62;
  v64 = sub_BD2C40(80, unk_3F10A10);
  v66 = (__int64)v64;
  if ( v64 )
    sub_B4D3C0((__int64)v64, v88, v51, 0, v63, v65, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v66,
    v91,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v67 = 16LL * *(unsigned int *)(a2 + 8);
  v68 = *(_QWORD *)a2;
  v69 = v68 + v67;
  while ( v69 != v68 )
  {
    v70 = *(_QWORD *)(v68 + 8);
    v71 = *(_DWORD *)v68;
    v68 += 16;
    sub_B99FD0(v66, v71, v70);
  }
}
