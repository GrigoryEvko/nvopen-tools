// Function: sub_10E7930
// Address: 0x10e7930
//
__int64 __fastcall sub_10E7930(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  bool v7; // bl
  int v8; // eax
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // rdx
  char v22; // al
  char v23; // al
  int v24; // eax
  __int64 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r15
  unsigned __int8 *v29; // r14
  __int64 v30; // rax
  __int64 v32; // rdx
  int v33; // r15d
  __int64 v34; // r15
  __int64 v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // [rsp-10h] [rbp-D0h]
  __int64 v39; // [rsp-8h] [rbp-C8h]
  __int64 v40; // [rsp+0h] [rbp-C0h]
  __int64 v41; // [rsp+0h] [rbp-C0h]
  __int64 v42; // [rsp+8h] [rbp-B8h]
  __int64 v43; // [rsp+8h] [rbp-B8h]
  __int64 v44; // [rsp+8h] [rbp-B8h]
  int v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+10h] [rbp-B0h]
  __int64 v47; // [rsp+10h] [rbp-B0h]
  __int64 v48; // [rsp+10h] [rbp-B0h]
  unsigned int v49; // [rsp+10h] [rbp-B0h]
  __int64 v50; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v51; // [rsp+28h] [rbp-98h] BYREF
  char v52[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v53; // [rsp+50h] [rbp-70h]
  _QWORD *v54; // [rsp+60h] [rbp-60h] BYREF
  __int64 v55; // [rsp+68h] [rbp-58h]
  __int16 v56; // [rsp+80h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 4);
  v50 = 0;
  v51 = 0;
  v4 = v3 & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v6 = *(_QWORD *)(a2 - 32 * v4);
  v7 = sub_B5B640(a2);
  v8 = sub_B5B5E0(a2);
  v9 = sub_1117800(a1, v8, v7, v6, v5, a2, (__int64)&v50, (__int64)&v51);
  v13 = v38;
  v14 = v39;
  if ( v9 )
    return sub_10DF260(a2, v50, v51);
  v15 = *(_QWORD *)(a2 + 16);
  if ( !v15 )
    return 0;
  while ( 1 )
  {
    v16 = *(_QWORD *)(v15 + 24);
    if ( *(_BYTE *)v16 == 93 && *(_DWORD *)(v16 + 80) == 1 )
    {
      v45 = **(_DWORD **)(v16 + 72);
      if ( v45 == 1 )
      {
        v17 = *(_QWORD *)(a1 + 64);
        if ( !*(_BYTE *)(v17 + 192) )
          sub_CFDFC0(*(_QWORD *)(a1 + 64), v14, v10, v13, v11, (__int64 *)v12);
        v18 = *(unsigned int *)(v17 + 184);
        if ( (_DWORD)v18 )
        {
          v14 = *(_QWORD *)(v17 + 168);
          v10 = ((_DWORD)v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v13 = v14 + 88 * v10;
          v11 = *(_QWORD *)(v13 + 24);
          if ( v16 != v11 )
          {
            while ( v11 != -4096 )
            {
              v12 = (unsigned int)(v45 + 1);
              v10 = ((_DWORD)v18 - 1) & (unsigned int)(v45 + v10);
              v13 = v14 + 88LL * (unsigned int)v10;
              v11 = *(_QWORD *)(v13 + 24);
              if ( v16 == v11 )
                goto LABEL_12;
              ++v45;
            }
            goto LABEL_4;
          }
LABEL_12:
          v10 = 5 * v18;
          if ( v13 != v14 + 88 * v18 )
          {
            v11 = *(_QWORD *)(v13 + 40);
            v19 = v11 + 32LL * *(unsigned int *)(v13 + 48);
            if ( v19 != v11 )
              break;
          }
        }
      }
    }
LABEL_4:
    v15 = *(_QWORD *)(v15 + 8);
    if ( !v15 )
      return 0;
  }
  while ( 1 )
  {
    v12 = *(_QWORD *)(v11 + 16);
    if ( v12 )
    {
      v54 = 0;
      v55 = v16;
      v10 = *(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v10 == 59 )
      {
        v40 = v12;
        v42 = v11;
        v46 = *(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
        v20 = sub_995B10(&v54, *(_QWORD *)(v10 - 64));
        v21 = v46;
        v11 = v42;
        v12 = v40;
        v14 = *(_QWORD *)(v46 - 32);
        if ( v20 )
        {
          if ( v14 == v55 )
            goto LABEL_22;
        }
        v47 = v42;
        v41 = v21;
        v43 = v12;
        v22 = sub_995B10(&v54, v14);
        v11 = v47;
        if ( v22 )
        {
          v10 = v41;
          v12 = v43;
          if ( *(_QWORD *)(v41 - 64) == v55 )
          {
LABEL_22:
            v14 = a2;
            v48 = v11;
            v23 = sub_98CF40(v12, a2, 0, 1);
            v11 = v48;
            if ( v23 )
              break;
          }
        }
      }
    }
    v11 += 32;
    if ( v11 == v19 )
      goto LABEL_4;
  }
  v24 = *(_DWORD *)(a2 + 4);
  v25 = *(__int64 **)(a1 + 32);
  v53 = 257;
  v26 = v24 & 0x7FFFFFF;
  v27 = *(_QWORD *)(a2 - 32 * v26);
  v28 = *(_QWORD *)(a2 + 32 * (1 - v26));
  v44 = v27;
  v49 = sub_B5B5E0(a2);
  v29 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v25[10] + 16LL))(
                             v25[10],
                             v49,
                             v27,
                             v28);
  if ( !v29 )
  {
    v56 = 257;
    v29 = (unsigned __int8 *)sub_B504D0(v49, v44, v28, (__int64)&v54, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v29) )
    {
      v32 = v25[12];
      v33 = *((_DWORD *)v25 + 26);
      if ( v32 )
        sub_B99FD0((__int64)v29, 3u, v32);
      sub_B45150((__int64)v29, v33);
    }
    (*(void (__fastcall **)(__int64, unsigned __int8 *, char *, __int64, __int64))(*(_QWORD *)v25[11] + 16LL))(
      v25[11],
      v29,
      v52,
      v25[7],
      v25[8]);
    v34 = *v25 + 16LL * *((unsigned int *)v25 + 2);
    if ( *v25 != v34 )
    {
      v35 = *v25;
      do
      {
        v36 = *(_QWORD *)(v35 + 8);
        v37 = *(_DWORD *)v35;
        v35 += 16;
        sub_B99FD0((__int64)v29, v37, v36);
      }
      while ( v34 != v35 );
    }
  }
  sub_BD6B90(v29, (unsigned __int8 *)a2);
  if ( *v29 > 0x1Cu )
  {
    if ( sub_B5B640(a2) )
      sub_B44850(v29, 1);
    else
      sub_B447F0(v29, 1);
  }
  v30 = sub_AD6450(*(_QWORD *)(v16 + 8));
  return sub_10DF260(a2, (__int64)v29, v30);
}
