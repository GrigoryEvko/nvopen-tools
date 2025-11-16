// Function: sub_11B2210
// Address: 0x11b2210
//
__int64 __fastcall sub_11B2210(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  bool v11; // zf
  void *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  void *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  void *v19; // rcx
  __int64 v20; // r10
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r10
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // r10
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rbx
  int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // r12
  __int64 v41; // rbx
  __int64 v42; // rdx
  unsigned int v43; // esi
  void *v44; // [rsp+8h] [rbp-C8h]
  void *v45; // [rsp+8h] [rbp-C8h]
  __int64 v46; // [rsp+8h] [rbp-C8h]
  __int64 v47; // [rsp+10h] [rbp-C0h]
  __int64 v48; // [rsp+10h] [rbp-C0h]
  void *v49; // [rsp+10h] [rbp-C0h]
  __int64 v50; // [rsp+18h] [rbp-B8h]
  __int64 v51; // [rsp+18h] [rbp-B8h]
  __int64 v52; // [rsp+18h] [rbp-B8h]
  __int64 v53; // [rsp+18h] [rbp-B8h]
  __int64 v54; // [rsp+20h] [rbp-B0h]
  __int64 v55; // [rsp+20h] [rbp-B0h]
  __int64 v56; // [rsp+20h] [rbp-B0h]
  _QWORD *v57; // [rsp+20h] [rbp-B0h]
  __int64 v58; // [rsp+20h] [rbp-B0h]
  __int64 v59; // [rsp+20h] [rbp-B0h]
  char v60; // [rsp+28h] [rbp-A8h]
  __int64 v61; // [rsp+28h] [rbp-A8h]
  __int64 v62; // [rsp+28h] [rbp-A8h]
  __int64 v63; // [rsp+28h] [rbp-A8h]
  __int64 v64; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+38h] [rbp-98h] BYREF
  __int64 v66[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v67; // [rsp+60h] [rbp-70h]
  __int64 *v68; // [rsp+70h] [rbp-60h] BYREF
  int v69; // [rsp+78h] [rbp-58h]
  int v70; // [rsp+80h] [rbp-50h]
  __int64 *v71; // [rsp+88h] [rbp-48h]
  __int16 v72; // [rsp+90h] [rbp-40h]

  v2 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v2 <= 0x1Cu )
    return 0;
  v3 = a1;
  v68 = &v64;
  v69 = 170;
  v70 = 0;
  v71 = &v64;
  if ( !(unsigned __int8)sub_11B1D40((__int64)&v68, v2) )
    return 0;
  v4 = *(_QWORD *)(a1 - 32);
  v60 = *(_BYTE *)v2;
  v5 = *(_QWORD *)(v2 + 16);
  if ( !v5 )
  {
    v6 = *(_BYTE *)v4;
LABEL_5:
    if ( v6 > 0x1Cu )
    {
      v69 = 170;
      v68 = &v65;
      v70 = 0;
      v71 = &v65;
      if ( (unsigned __int8)sub_11B1D40((__int64)&v68, v4) )
      {
        if ( *(_BYTE *)v4 == *(_BYTE *)v2
          && ((v7 = *(_QWORD *)(v2 + 16)) != 0 && !*(_QWORD *)(v7 + 8)
           || (v8 = *(_QWORD *)(v4 + 16)) != 0 && !*(_QWORD *)(v8 + 8)) )
        {
          v15 = a2[10];
          v16 = *(void **)(v3 + 72);
          v67 = 257;
          v45 = v16;
          v55 = v64;
          v48 = *(unsigned int *)(v3 + 80);
          v51 = v65;
          v17 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 112LL))(v15, v64);
          v18 = v51;
          v19 = v45;
          v20 = v17;
          if ( !v17 )
          {
            v46 = v48;
            v49 = v19;
            v52 = v55;
            v56 = v18;
            v72 = 257;
            v26 = sub_BD2C40(112, unk_3F1FE60);
            v27 = v26;
            if ( v26 )
            {
              v28 = v56;
              v57 = v26;
              sub_B4E9E0((__int64)v26, v52, v28, v49, v46, (__int64)&v68, 0, 0);
              v27 = v57;
            }
            v58 = (__int64)v27;
            (*(void (__fastcall **)(__int64, _QWORD *, __int64 *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
              a2[11],
              v27,
              v66,
              a2[7],
              a2[8]);
            v20 = v58;
            if ( *a2 != *a2 + 16LL * *((unsigned int *)a2 + 2) )
            {
              v59 = v2;
              v29 = v20;
              v53 = v3;
              v30 = *a2;
              v31 = *a2 + 16LL * *((unsigned int *)a2 + 2);
              do
              {
                v32 = *(_QWORD *)(v30 + 8);
                v33 = *(_DWORD *)v30;
                v30 += 16;
                sub_B99FD0(v29, v33, v32);
              }
              while ( v31 != v30 );
              v20 = v29;
              v3 = v53;
              v2 = v59;
            }
          }
          if ( v60 == 41 )
          {
            v72 = 257;
            v9 = sub_B50340(12, v20, (__int64)&v68, 0, 0);
          }
          else
          {
            v61 = v20;
            v68 = *(__int64 **)(v3 + 8);
            v21 = (__int64 *)sub_B43CA0(v3);
            v22 = sub_B6E160(v21, 0xAAu, (__int64)&v68, 1);
            v72 = 257;
            v23 = v22;
            v66[0] = v61;
            v24 = 0;
            if ( v22 )
              v24 = *(_QWORD *)(v22 + 24);
            v62 = v24;
            v25 = sub_BD2CC0(88, 2u);
            v9 = (__int64)v25;
            if ( v25 )
            {
              sub_B44260((__int64)v25, **(_QWORD **)(v62 + 16), 56, 2u, 0, 0);
              *(_QWORD *)(v9 + 72) = 0;
              sub_B4A290(v9, v62, v23, v66, 1, (__int64)&v68, 0, 0);
            }
          }
          sub_B45260((unsigned __int8 *)v9, v2, 1);
          sub_B45560((unsigned __int8 *)v9, v4);
          return v9;
        }
      }
    }
    return 0;
  }
  v11 = *(_QWORD *)(v5 + 8) == 0;
  v6 = *(_BYTE *)v4;
  if ( !v11 || v6 != 13 )
    goto LABEL_5;
  v12 = *(void **)(a1 + 72);
  v13 = *(unsigned int *)(a1 + 80);
  v67 = 257;
  v44 = v12;
  v50 = v64;
  v54 = v13;
  v47 = sub_ACADE0(*(__int64 ***)(v64 + 8));
  v14 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, void *, __int64))(*(_QWORD *)a2[10] + 112LL))(
          a2[10],
          v50,
          v47,
          v12,
          v54);
  if ( !v14 )
  {
    v72 = 257;
    v39 = sub_BD2C40(112, unk_3F1FE60);
    v14 = (__int64)v39;
    if ( v39 )
      sub_B4E9E0((__int64)v39, v50, v47, v44, v54, (__int64)&v68, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v14,
      v66,
      a2[7],
      a2[8]);
    if ( *a2 != *a2 + 16LL * *((unsigned int *)a2 + 2) )
    {
      v40 = *a2;
      v41 = *a2 + 16LL * *((unsigned int *)a2 + 2);
      do
      {
        v42 = *(_QWORD *)(v40 + 8);
        v43 = *(_DWORD *)v40;
        v40 += 16;
        sub_B99FD0(v14, v43, v42);
      }
      while ( v41 != v40 );
      v3 = a1;
    }
  }
  if ( v60 == 41 )
  {
    v72 = 257;
    v9 = sub_B50340(12, v14, (__int64)&v68, 0, 0);
    sub_B45260((unsigned __int8 *)v9, v2, 1);
  }
  else
  {
    v68 = *(__int64 **)(v3 + 8);
    v34 = (__int64 *)sub_B43CA0(v3);
    v35 = sub_B6E160(v34, 0xAAu, (__int64)&v68, 1);
    v66[0] = v14;
    v36 = 0;
    v72 = 257;
    v37 = v35;
    if ( v35 )
      v36 = *(_QWORD *)(v35 + 24);
    v9 = (__int64)sub_BD2CC0(88, 2u);
    if ( v9 )
    {
      v63 = v9;
      sub_B44260(v9, **(_QWORD **)(v36 + 16), 56, 2u, 0, 0);
      *(_QWORD *)(v9 + 72) = 0;
      sub_B4A290(v9, v36, v37, v66, 1, (__int64)&v68, 0, 0);
    }
    else
    {
      v63 = 0;
    }
    v38 = sub_B45210(v2);
    sub_B45150(v63, v38);
  }
  return v9;
}
