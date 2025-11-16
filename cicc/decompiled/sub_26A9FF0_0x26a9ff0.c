// Function: sub_26A9FF0
// Address: 0x26a9ff0
//
__int64 __fastcall sub_26A9FF0(__int64 *a1, unsigned __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v8; // edx
  __int64 *v9; // r13
  __int64 v10; // r8
  __int64 v11; // r12
  char v12; // al
  __int64 result; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  unsigned __int64 v20; // rax
  __int64 v21; // r15
  int v22; // eax
  __int64 v23; // rdi
  bool v24; // zf
  _BYTE *v25; // rax
  char v26; // dl
  int v27; // r10d
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rax
  void *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rax
  void *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rax
  void *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rax
  void *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  unsigned int v63; // eax
  void *v64; // rax
  size_t v65; // rdx
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  _QWORD *v68; // rdi
  _QWORD *v69; // rsi
  int v70; // eax
  __int64 v71; // rsi
  int v72; // edx
  unsigned int v73; // eax
  __int64 v74; // rcx
  int v75; // edi
  __int64 v76; // [rsp-10h] [rbp-60h]
  __int64 v77; // [rsp+8h] [rbp-48h]
  __int64 v78; // [rsp+10h] [rbp-40h] BYREF
  __int64 v79; // [rsp+18h] [rbp-38h]

  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 34560);
  v6 = *(unsigned int *)(v4 + 34576);
  if ( !(_DWORD)v6 )
    goto LABEL_17;
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v27 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v27 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      ++v27;
    }
    goto LABEL_17;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v5 + 16 * v6) )
  {
LABEL_17:
    sub_250D230((unsigned __int64 *)&v78, a2, 4, 0);
    v28 = sub_26A73D0(a1[1], v78, v79, a1[2], 0, 1);
    v29 = v28;
    if ( v28 )
    {
      v30 = a1[2];
      v32 = (unsigned int)sub_266F260(v30 + 88, v28 + 88);
      result = 1;
      if ( !(_BYTE)v32 )
      {
        v34 = v29 + 120;
        *(_BYTE *)(v30 + 96) = *(_BYTE *)(v29 + 96);
        *(_BYTE *)(v30 + 112) = *(_BYTE *)(v29 + 112);
        *(_BYTE *)(v30 + 113) = *(_BYTE *)(v29 + 113);
        if ( v29 + 120 != v30 + 120 )
        {
          sub_C7D6A0(*(_QWORD *)(v30 + 128), 8LL * *(unsigned int *)(v30 + 144), 8);
          v35 = *(unsigned int *)(v29 + 144);
          *(_DWORD *)(v30 + 144) = v35;
          if ( (_DWORD)v35 )
          {
            v36 = (void *)sub_C7D670(8 * v35, 8);
            v37 = *(unsigned int *)(v30 + 144);
            *(_QWORD *)(v30 + 128) = v36;
            *(_DWORD *)(v30 + 136) = *(_DWORD *)(v29 + 136);
            *(_DWORD *)(v30 + 140) = *(_DWORD *)(v29 + 140);
            memcpy(v36, *(const void **)(v29 + 128), 8 * v37);
          }
          else
          {
            *(_QWORD *)(v30 + 128) = 0;
            *(_QWORD *)(v30 + 136) = 0;
          }
        }
        sub_266EB10(v30 + 152, v29 + 152, v34, v31, v32, v33);
        v41 = v29 + 184;
        *(_BYTE *)(v30 + 176) = *(_BYTE *)(v29 + 176);
        *(_BYTE *)(v30 + 177) = *(_BYTE *)(v29 + 177);
        if ( v29 + 184 != v30 + 184 )
        {
          sub_C7D6A0(*(_QWORD *)(v30 + 192), 8LL * *(unsigned int *)(v30 + 208), 8);
          v42 = *(unsigned int *)(v29 + 208);
          *(_DWORD *)(v30 + 208) = v42;
          if ( (_DWORD)v42 )
          {
            v43 = (void *)sub_C7D670(8 * v42, 8);
            v44 = *(unsigned int *)(v30 + 208);
            *(_QWORD *)(v30 + 192) = v43;
            *(_DWORD *)(v30 + 200) = *(_DWORD *)(v29 + 200);
            *(_DWORD *)(v30 + 204) = *(_DWORD *)(v29 + 204);
            memcpy(v43, *(const void **)(v29 + 192), 8 * v44);
          }
          else
          {
            *(_QWORD *)(v30 + 192) = 0;
            *(_QWORD *)(v30 + 200) = 0;
          }
        }
        sub_266EB10(v30 + 216, v29 + 216, v41, v38, v39, v40);
        v48 = v29 + 248;
        *(_BYTE *)(v30 + 240) = *(_BYTE *)(v29 + 240);
        *(_BYTE *)(v30 + 241) = *(_BYTE *)(v29 + 241);
        if ( v29 + 248 != v30 + 248 )
        {
          sub_C7D6A0(*(_QWORD *)(v30 + 256), 8LL * *(unsigned int *)(v30 + 272), 8);
          v49 = *(unsigned int *)(v29 + 272);
          *(_DWORD *)(v30 + 272) = v49;
          if ( (_DWORD)v49 )
          {
            v50 = (void *)sub_C7D670(8 * v49, 8);
            v51 = *(unsigned int *)(v30 + 272);
            *(_QWORD *)(v30 + 256) = v50;
            *(_DWORD *)(v30 + 264) = *(_DWORD *)(v29 + 264);
            *(_DWORD *)(v30 + 268) = *(_DWORD *)(v29 + 268);
            memcpy(v50, *(const void **)(v29 + 256), 8 * v51);
          }
          else
          {
            *(_QWORD *)(v30 + 256) = 0;
            *(_QWORD *)(v30 + 264) = 0;
          }
        }
        sub_266EA30(v30 + 280, v29 + 280, v48, v45, v46, v47);
        v55 = v29 + 344;
        *(_QWORD *)(v30 + 296) = *(_QWORD *)(v29 + 296);
        *(_QWORD *)(v30 + 304) = *(_QWORD *)(v29 + 304);
        *(_QWORD *)(v30 + 312) = *(_QWORD *)(v29 + 312);
        *(_BYTE *)(v30 + 320) = *(_BYTE *)(v29 + 320);
        *(_BYTE *)(v30 + 336) = *(_BYTE *)(v29 + 336);
        *(_BYTE *)(v30 + 337) = *(_BYTE *)(v29 + 337);
        if ( v29 + 344 != v30 + 344 )
        {
          sub_C7D6A0(*(_QWORD *)(v30 + 352), 8LL * *(unsigned int *)(v30 + 368), 8);
          v56 = *(unsigned int *)(v29 + 368);
          *(_DWORD *)(v30 + 368) = v56;
          if ( (_DWORD)v56 )
          {
            v57 = (void *)sub_C7D670(8 * v56, 8);
            v58 = *(unsigned int *)(v30 + 368);
            *(_QWORD *)(v30 + 352) = v57;
            *(_DWORD *)(v30 + 360) = *(_DWORD *)(v29 + 360);
            *(_DWORD *)(v30 + 364) = *(_DWORD *)(v29 + 364);
            memcpy(v57, *(const void **)(v29 + 352), 8 * v58);
          }
          else
          {
            *(_QWORD *)(v30 + 352) = 0;
            *(_QWORD *)(v30 + 360) = 0;
          }
        }
        sub_266E950(v30 + 376, v29 + 376, v55, v52, v53, v54);
        v62 = v29 + 408;
        *(_BYTE *)(v30 + 400) = *(_BYTE *)(v29 + 400);
        *(_BYTE *)(v30 + 401) = *(_BYTE *)(v29 + 401);
        if ( v29 + 408 != v30 + 408 )
        {
          sub_C7D6A0(*(_QWORD *)(v30 + 416), *(unsigned int *)(v30 + 432), 1);
          v63 = *(_DWORD *)(v29 + 432);
          *(_DWORD *)(v30 + 432) = v63;
          if ( v63 )
          {
            v64 = (void *)sub_C7D670(v63, 1);
            v65 = *(unsigned int *)(v30 + 432);
            *(_QWORD *)(v30 + 416) = v64;
            *(_DWORD *)(v30 + 424) = *(_DWORD *)(v29 + 424);
            *(_DWORD *)(v30 + 428) = *(_DWORD *)(v29 + 428);
            memcpy(v64, *(const void **)(v29 + 416), v65);
          }
          else
          {
            *(_QWORD *)(v30 + 416) = 0;
            *(_QWORD *)(v30 + 424) = 0;
          }
        }
        sub_266E880(v30 + 440, v29 + 440, v62, v59, v60, v61);
        *(_BYTE *)(v30 + 464) = *(_BYTE *)(v29 + 464);
        return 0;
      }
      return result;
    }
    v25 = (_BYTE *)a1[2];
LABEL_14:
    v26 = v25[400];
    v25[96] = 1;
    v25[464] = 1;
    v25[401] = v26;
    v25[337] = v25[336];
    v25[241] = v25[240];
    v25[113] = v25[112];
    v25[177] = v25[176];
    return 0;
  }
  v11 = a1[2];
  if ( a3 > 1 )
  {
    v12 = *(_BYTE *)(v11 + 400);
    *(_BYTE *)(v11 + 96) = 1;
    *(_BYTE *)(v11 + 464) = 1;
    *(_BYTE *)(v11 + 401) = v12;
    *(_BYTE *)(v11 + 337) = *(_BYTE *)(v11 + 336);
    *(_BYTE *)(v11 + 241) = *(_BYTE *)(v11 + 240);
    *(_BYTE *)(v11 + 113) = *(_BYTE *)(v11 + 112);
    *(_BYTE *)(v11 + 177) = *(_BYTE *)(v11 + 176);
    return 0;
  }
  v14 = sub_250D070((_QWORD *)(v11 + 72));
  v15 = a1[1];
  v16 = v14;
  if ( *((_DWORD *)v9 + 2) == 158 )
  {
    v24 = (unsigned __int8)sub_26A95D0(v11, a1[1], v14) == 0;
    v25 = (_BYTE *)a1[2];
    if ( !v24 )
      return (unsigned __int8)sub_266F260(a1[3], (__int64)(v25 + 88));
    goto LABEL_14;
  }
  v17 = sub_B491C0(v14);
  sub_250D230((unsigned __int64 *)&v78, v17, 4, 0);
  v18 = sub_269DF00(v15, v78, v79, a1[2], 1, 0, 1);
  v19 = a1[1];
  v77 = v18;
  v20 = sub_B491C0(v16);
  sub_250D230((unsigned __int64 *)&v78, v20, 4, 0);
  v21 = sub_269D460(v19, v78, v79, a1[2], 1);
  v22 = *((_DWORD *)v9 + 2);
  if ( v22 == 180 )
  {
    if ( v77 && (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v77 + 112LL))(v77, v16, v76) )
      return 0;
    if ( v21 && *(_BYTE *)(v21 + 97) )
    {
      v78 = v16;
      if ( *(_DWORD *)(v21 + 120) )
      {
        v70 = *(_DWORD *)(v21 + 128);
        v71 = *(_QWORD *)(v21 + 112);
        if ( v70 )
        {
          v72 = v70 - 1;
          v73 = (v70 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v74 = *(_QWORD *)(v71 + 8LL * v73);
          if ( v16 == v74 )
            return 0;
          v75 = 1;
          while ( v74 != -4096 )
          {
            v73 = v72 & (v75 + v73);
            v74 = *(_QWORD *)(v71 + 8LL * v73);
            if ( v16 == v74 )
              return 0;
            ++v75;
          }
        }
      }
      else
      {
        v68 = *(_QWORD **)(v21 + 136);
        v69 = &v68[*(unsigned int *)(v21 + 144)];
        if ( v69 != sub_266E350(v68, (__int64)v69, &v78) )
          return 0;
      }
    }
    goto LABEL_11;
  }
  if ( v22 != 181 )
  {
    *(_BYTE *)(a1[2] + 241) = *(_BYTE *)(a1[2] + 240);
    goto LABEL_11;
  }
  if ( v77 && (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v77 + 120LL))(v77, v16, v76) )
    return 0;
  if ( !v21 || !*(_BYTE *)(v21 + 97) )
    goto LABEL_11;
  if ( !*(_BYTE *)(v21 + 212) )
  {
    if ( sub_C8CA60(v21 + 184, v16) )
      return 0;
    goto LABEL_11;
  }
  v66 = *(_QWORD **)(v21 + 192);
  v67 = &v66[*(unsigned int *)(v21 + 204)];
  if ( v66 == v67 )
  {
LABEL_11:
    v23 = a1[2];
    v78 = v16;
    sub_269CCD0(v23 + 248, &v78);
    return 0;
  }
  while ( v16 != *v66 )
  {
    if ( v67 == ++v66 )
      goto LABEL_11;
  }
  return 0;
}
