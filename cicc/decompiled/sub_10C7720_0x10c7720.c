// Function: sub_10C7720
// Address: 0x10c7720
//
_QWORD *__fastcall sub_10C7720(_QWORD *a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r10
  __int64 v15; // r15
  __int64 *v16; // r14
  int *v17; // rax
  int v18; // ebx
  __int64 v19; // r13
  __int64 v20; // r14
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // r10
  __int64 v24; // rdx
  int v25; // ecx
  int v26; // eax
  _QWORD *v27; // rdi
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r10
  __int64 v32; // rbx
  __int64 i; // r15
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rdx
  int v37; // r15d
  __int64 v38; // rbx
  __int64 v39; // r14
  __int64 v40; // rdx
  unsigned int v41; // esi
  _QWORD *v42; // [rsp+0h] [rbp-B0h]
  __int64 v43; // [rsp+8h] [rbp-A8h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+18h] [rbp-98h]
  _BYTE v50[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v51; // [rsp+40h] [rbp-70h]
  __int64 *v52; // [rsp+50h] [rbp-60h] BYREF
  __int64 v53[3]; // [rsp+58h] [rbp-58h] BYREF
  __int16 v54; // [rsp+70h] [rbp-40h]

  v6 = sub_BCB060(*(_QWORD *)(a2 + 8));
  v52 = &v48;
  v53[0] = (unsigned int)(v6 - 1);
  v7 = *(_QWORD *)(a2 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 55 )
    return 0;
  if ( !*(_QWORD *)(a2 - 64) )
    return 0;
  v10 = *(_QWORD *)(a2 - 32);
  v48 = *(_QWORD *)(a2 - 64);
  if ( !sub_F17ED0(v53, v10) )
    return 0;
  v11 = *(_QWORD *)(a3 + 16);
  if ( !v11 || *(_QWORD *)(v11 + 8) || *(_BYTE *)a3 != 68 || **(_BYTE **)(a3 - 32) != 82 )
    return 0;
  v12 = *(__int64 **)(*a1 + 32LL);
  v51 = 257;
  v13 = sub_AD6530(*(_QWORD *)(v48 + 8), v10);
  v43 = v48;
  v14 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v12[10] + 56LL))(
          v12[10],
          40,
          v48,
          v13);
  if ( !v14 )
  {
    v54 = 257;
    v22 = sub_BD2C40(72, unk_3F10FD0);
    v23 = v22;
    if ( v22 )
    {
      v42 = v22;
      v24 = *(_QWORD *)(v43 + 8);
      v25 = *(unsigned __int8 *)(v24 + 8);
      if ( (unsigned int)(v25 - 17) > 1 )
      {
        v29 = sub_BCB2A0(*(_QWORD **)v24);
        v31 = (__int64)v42;
        v30 = v43;
      }
      else
      {
        v26 = *(_DWORD *)(v24 + 32);
        v27 = *(_QWORD **)v24;
        BYTE4(v49) = (_BYTE)v25 == 18;
        LODWORD(v49) = v26;
        v28 = (__int64 *)sub_BCB2A0(v27);
        v29 = sub_BCE1B0(v28, v49);
        v30 = v43;
        v31 = (__int64)v42;
      }
      v45 = v31;
      sub_B523C0(v31, v29, 53, 40, v30, v13, (__int64)&v52, 0, 0, 0);
      v23 = (_QWORD *)v45;
    }
    v46 = (__int64)v23;
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE *, __int64, __int64))(*(_QWORD *)v12[11] + 16LL))(
      v12[11],
      v23,
      v50,
      v12[7],
      v12[8]);
    v32 = *v12;
    v14 = v46;
    for ( i = *v12 + 16LL * *((unsigned int *)v12 + 2); i != v32; v14 = v47 )
    {
      v34 = *(_QWORD *)(v32 + 8);
      v35 = *(_DWORD *)v32;
      v32 += 16;
      v47 = v14;
      sub_B99FD0(v14, v35, v34);
    }
  }
  v15 = *(_QWORD *)(a3 - 32);
  v44 = v14;
  v16 = *(__int64 **)(*a1 + 32LL);
  v17 = (int *)a1[1];
  v51 = 257;
  v18 = *v17;
  v19 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v16[10] + 16LL))(
          v16[10],
          (unsigned int)*v17,
          v14,
          v15);
  if ( !v19 )
  {
    v54 = 257;
    v19 = sub_B504D0(v18, v44, v15, (__int64)&v52, 0, 0);
    if ( (unsigned __int8)sub_920620(v19) )
    {
      v36 = v16[12];
      v37 = *((_DWORD *)v16 + 26);
      if ( v36 )
        sub_B99FD0(v19, 3u, v36);
      sub_B45150(v19, v37);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
      v16[11],
      v19,
      v50,
      v16[7],
      v16[8]);
    v38 = *v16;
    v39 = *v16 + 16LL * *((unsigned int *)v16 + 2);
    while ( v39 != v38 )
    {
      v40 = *(_QWORD *)(v38 + 8);
      v41 = *(_DWORD *)v38;
      v38 += 16;
      sub_B99FD0(v19, v41, v40);
    }
  }
  v20 = *(_QWORD *)(a2 + 8);
  v54 = 257;
  v21 = sub_BD2C40(72, unk_3F10A14);
  v8 = v21;
  if ( v21 )
    sub_B515B0((__int64)v21, v19, v20, (__int64)&v52, 0, 0);
  return v8;
}
