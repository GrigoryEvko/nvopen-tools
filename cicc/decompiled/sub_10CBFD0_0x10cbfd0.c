// Function: sub_10CBFD0
// Address: 0x10cbfd0
//
__int64 __fastcall sub_10CBFD0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r13
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rdi
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // r15
  unsigned int **v19; // rdi
  __int64 v20; // r13
  __int64 v21; // r10
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdx
  int v26; // r13d
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v32; // rdx
  int v33; // r8d
  unsigned int *v34; // rax
  __int64 v35; // r15
  unsigned int *v36; // r15
  unsigned int *v37; // rbx
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // r9
  char v41; // al
  __int64 v42; // r10
  __int64 v43; // rdx
  int v44; // r14d
  unsigned int *v45; // rax
  __int64 v46; // r15
  unsigned int *v47; // rbx
  __int64 v48; // r14
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // r14
  unsigned int *v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // [rsp+8h] [rbp-C8h]
  int v60; // [rsp+18h] [rbp-B8h]
  _QWORD *v61; // [rsp+18h] [rbp-B8h]
  __int64 v63; // [rsp+20h] [rbp-B0h]
  _QWORD *v64; // [rsp+20h] [rbp-B0h]
  __int64 v66; // [rsp+28h] [rbp-A8h]
  __int64 v67; // [rsp+28h] [rbp-A8h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  _QWORD *v69; // [rsp+28h] [rbp-A8h]
  __int64 v70; // [rsp+28h] [rbp-A8h]
  __int64 v71; // [rsp+28h] [rbp-A8h]
  __int64 v72; // [rsp+38h] [rbp-98h]
  _BYTE v73[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v74; // [rsp+60h] [rbp-70h]
  _QWORD v75[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v76; // [rsp+90h] [rbp-40h]

  v6 = a2;
  v7 = a1;
  v8 = a4;
  v58 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)a2 == 78 )
  {
    v56 = *(_QWORD *)(a2 + 16);
    if ( v56 )
    {
      if ( !*(_QWORD *)(v56 + 8) )
        v6 = *(_QWORD *)(a2 - 32);
    }
  }
  if ( *(_BYTE *)a4 == 78 )
  {
    v57 = *(_QWORD *)(a4 + 16);
    if ( v57 )
    {
      if ( !*(_QWORD *)(v57 + 8) )
        v8 = *(_QWORD *)(a4 - 32);
    }
  }
  v9 = sub_10CB120(a1, v6, v8, a6);
  v10 = v9;
  if ( v9 )
  {
    v11 = *(_QWORD *)(v9 + 8);
    v12 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    {
      v13 = *(_QWORD *)(v6 + 8);
      v14 = *(_DWORD *)(v11 + 32);
      v15 = sub_BCAE30(v13);
      v75[1] = v16;
      v75[0] = v15;
      v17 = (__int64 *)sub_BCD140(*(_QWORD **)(v7[4] + 72LL), (unsigned int)v15 / v14);
      BYTE4(v72) = *(_BYTE *)(v11 + 8) == 18;
      LODWORD(v72) = *(_DWORD *)(v11 + 32);
      v12 = sub_BCE1B0(v17, v72);
    }
    v18 = v7[4];
    v74 = 257;
    v19 = (unsigned int **)v18;
    if ( v12 == *(_QWORD *)(a3 + 8) )
    {
      v20 = a3;
    }
    else
    {
      v20 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v18 + 80) + 120LL))(
              *(_QWORD *)(v18 + 80),
              49,
              a3,
              v12);
      if ( !v20 )
      {
        v76 = 257;
        v20 = sub_B51D30(49, a3, v12, (__int64)v75, 0, 0);
        if ( (unsigned __int8)sub_920620(v20) )
        {
          v32 = *(_QWORD *)(v18 + 96);
          v33 = *(_DWORD *)(v18 + 104);
          if ( v32 )
          {
            v60 = *(_DWORD *)(v18 + 104);
            sub_B99FD0(v20, 3u, v32);
            v33 = v60;
          }
          sub_B45150(v20, v33);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v18 + 88) + 16LL))(
          *(_QWORD *)(v18 + 88),
          v20,
          v73,
          *(_QWORD *)(v18 + 56),
          *(_QWORD *)(v18 + 64));
        v34 = *(unsigned int **)v18;
        v35 = 4LL * *(unsigned int *)(v18 + 8);
        if ( v34 != &v34[v35] )
        {
          v61 = v7;
          v36 = &v34[v35];
          v37 = v34;
          do
          {
            v38 = *((_QWORD *)v37 + 1);
            v39 = *v37;
            v37 += 4;
            sub_B99FD0(v20, v39, v38);
          }
          while ( v36 != v37 );
          v7 = v61;
        }
        v18 = v7[4];
        v19 = (unsigned int **)v18;
        if ( !a6 )
          goto LABEL_10;
        goto LABEL_31;
      }
      v18 = v7[4];
      v19 = (unsigned int **)v18;
    }
    if ( !a6 )
    {
LABEL_10:
      v74 = 257;
      if ( v12 == *(_QWORD *)(a5 + 8) )
      {
        v21 = a5;
      }
      else
      {
        v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v18 + 80) + 120LL))(
                *(_QWORD *)(v18 + 80),
                49,
                a5,
                v12);
        if ( !v21 )
        {
          v76 = 257;
          v66 = sub_B51D30(49, a5, v12, (__int64)v75, 0, 0);
          v41 = sub_920620(v66);
          v42 = v66;
          if ( v41 )
          {
            v43 = *(_QWORD *)(v18 + 96);
            v44 = *(_DWORD *)(v18 + 104);
            if ( v43 )
            {
              sub_B99FD0(v66, 3u, v43);
              v42 = v66;
            }
            v67 = v42;
            sub_B45150(v42, v44);
            v42 = v67;
          }
          v68 = v42;
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v18 + 88) + 16LL))(
            *(_QWORD *)(v18 + 88),
            v42,
            v73,
            *(_QWORD *)(v18 + 56),
            *(_QWORD *)(v18 + 64));
          v45 = *(unsigned int **)v18;
          v21 = v68;
          if ( *(_QWORD *)v18 != *(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 8) )
          {
            v69 = v7;
            v46 = *(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 8);
            v47 = v45;
            v48 = v21;
            do
            {
              v49 = *((_QWORD *)v47 + 1);
              v50 = *v47;
              v47 += 4;
              sub_B99FD0(v48, v50, v49);
            }
            while ( (unsigned int *)v46 != v47 );
            v7 = v69;
            v21 = v48;
          }
        }
        v19 = (unsigned int **)v7[4];
      }
      v76 = 257;
      v22 = sub_B36550(v19, v10, v20, v21, (__int64)v75, 0);
      v23 = (__int64 *)v7[4];
      v74 = 257;
      v24 = v22;
      if ( v58 == *(_QWORD *)(v22 + 8) )
        return v22;
      v10 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v23[10] + 120LL))(v23[10], 49, v22);
      if ( !v10 )
      {
        v76 = 257;
        v10 = sub_B51D30(49, v24, v58, (__int64)v75, 0, 0);
        if ( (unsigned __int8)sub_920620(v10) )
        {
          v25 = v23[12];
          v26 = *((_DWORD *)v23 + 26);
          if ( v25 )
            sub_B99FD0(v10, 3u, v25);
          sub_B45150(v10, v26);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
          v23[11],
          v10,
          v73,
          v23[7],
          v23[8]);
        v27 = *v23;
        v28 = *v23 + 16LL * *((unsigned int *)v23 + 2);
        if ( *v23 != v28 )
        {
          do
          {
            v29 = *(_QWORD *)(v27 + 8);
            v30 = *(_DWORD *)v27;
            v27 += 16;
            sub_B99FD0(v10, v30, v29);
          }
          while ( v28 != v27 );
        }
      }
      return v10;
    }
LABEL_31:
    v74 = 257;
    v63 = sub_AD62B0(*(_QWORD *)(a5 + 8));
    v40 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v18 + 80) + 16LL))(
            *(_QWORD *)(v18 + 80),
            30,
            a5,
            v63);
    if ( !v40 )
    {
      v76 = 257;
      v70 = sub_B504D0(30, a5, v63, (__int64)v75, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v18 + 88) + 16LL))(
        *(_QWORD *)(v18 + 88),
        v70,
        v73,
        *(_QWORD *)(v18 + 56),
        *(_QWORD *)(v18 + 64));
      v40 = v70;
      if ( *(_QWORD *)v18 != *(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 8) )
      {
        v71 = v12;
        v51 = v40;
        v64 = v7;
        v52 = *(unsigned int **)v18;
        v53 = *(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 8);
        do
        {
          v54 = *((_QWORD *)v52 + 1);
          v55 = *v52;
          v52 += 4;
          sub_B99FD0(v51, v55, v54);
        }
        while ( (unsigned int *)v53 != v52 );
        v40 = v51;
        v7 = v64;
        v12 = v71;
      }
    }
    v18 = v7[4];
    a5 = v40;
    v19 = (unsigned int **)v18;
    goto LABEL_10;
  }
  return v10;
}
