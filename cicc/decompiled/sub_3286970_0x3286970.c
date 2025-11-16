// Function: sub_3286970
// Address: 0x3286970
//
__int64 __fastcall sub_3286970(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10)
{
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // ecx
  unsigned int v17; // edx
  unsigned __int64 v18; // rdi
  unsigned int v19; // ecx
  unsigned __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rax
  __int16 v23; // dx
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int128 v27; // rax
  __int128 *v28; // r14
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned __int16 *v31; // rax
  __int64 v32; // r15
  unsigned int v33; // ebx
  int v34; // eax
  unsigned int v35; // edx
  char v36; // al
  __int64 v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-A0h]
  __int64 v39; // [rsp+8h] [rbp-98h]
  unsigned __int64 v40; // [rsp+8h] [rbp-98h]
  unsigned int v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  char v43; // [rsp+10h] [rbp-90h]
  unsigned int v44; // [rsp+10h] [rbp-90h]
  __int128 v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+20h] [rbp-80h]
  unsigned int v47; // [rsp+20h] [rbp-80h]
  unsigned int v48; // [rsp+20h] [rbp-80h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  const void **v50; // [rsp+28h] [rbp-78h]
  __int64 v51; // [rsp+38h] [rbp-68h]
  unsigned __int64 v52; // [rsp+40h] [rbp-60h] BYREF
  __int64 v53; // [rsp+48h] [rbp-58h]
  __int64 v54; // [rsp+50h] [rbp-50h] BYREF
  __int64 v55; // [rsp+58h] [rbp-48h]
  __int64 v56; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v57; // [rsp+68h] [rbp-38h]

  if ( a6 != a2 || a5 != a1 )
  {
    if ( *(_DWORD *)(a5 + 24) != 216 )
      return 0;
    v12 = *(_QWORD *)(a5 + 40);
    if ( *(_QWORD *)v12 != a1 || *(_DWORD *)(v12 + 8) != a2 )
      return 0;
  }
  if ( *(_DWORD *)(a1 + 24) != 227 )
    return 0;
  if ( a9 != 12 )
    return 0;
  v13 = sub_33DFBC0(a3, a4, 0, 0);
  v14 = sub_33DFBC0(a7, a8, 0, 0);
  if ( !v13 || !v14 )
    return 0;
  v15 = *(_QWORD *)(v13 + 96);
  v42 = *(_QWORD *)(v14 + 96);
  v50 = (const void **)(v15 + 24);
  LODWORD(v55) = *(_DWORD *)(v15 + 32);
  if ( (unsigned int)v55 > 0x40 )
  {
    v49 = v15;
    sub_C43780((__int64)&v54, v50);
    v15 = v49;
  }
  else
  {
    v54 = *(_QWORD *)(v15 + 24);
  }
  v46 = v15;
  sub_C46A40((__int64)&v54, 1);
  v16 = v55;
  LODWORD(v55) = 0;
  v57 = v16;
  v56 = v54;
  if ( v16 <= 0x40 )
  {
    if ( v54 )
    {
      if ( (v54 & (v54 - 1)) == 0 )
      {
        v17 = *(_DWORD *)(v46 + 32);
        if ( *(_DWORD *)(v42 + 32) <= v17 )
        {
          v39 = v54;
          v47 = v16;
          sub_C449B0((__int64)&v52, (const void **)(v42 + 24), v17);
          v43 = sub_AAD8B0((__int64)v50, &v52) ^ 1;
          if ( (unsigned int)v53 <= 0x40 )
            goto LABEL_22;
          v18 = v52;
          v19 = v47;
          v20 = v39;
          if ( !v52 )
            goto LABEL_22;
          goto LABEL_19;
        }
      }
    }
    return 0;
  }
  v38 = v54;
  v41 = v16;
  v34 = sub_C44630((__int64)&v56);
  v20 = v38;
  if ( v34 == 1 && (v35 = *(_DWORD *)(v46 + 32), *(_DWORD *)(v42 + 32) <= v35) )
  {
    sub_C449B0((__int64)&v52, (const void **)(v42 + 24), v35);
    v36 = sub_AAD8B0((__int64)v50, &v52);
    v19 = v41;
    v20 = v38;
    v43 = v36 ^ 1;
    if ( (unsigned int)v53 > 0x40 )
    {
      v18 = v52;
      if ( v52 )
      {
LABEL_19:
        v40 = v20;
        v48 = v19;
        j_j___libc_free_0_0(v18);
        v20 = v40;
        if ( v48 <= 0x40 )
          goto LABEL_22;
      }
    }
  }
  else
  {
    v43 = 1;
  }
  if ( v20 )
    j_j___libc_free_0_0(v20);
LABEL_22:
  if ( (unsigned int)v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( v43 )
    return 0;
  sub_9865C0((__int64)&v54, (__int64)v50);
  sub_C46A40((__int64)&v54, 1);
  v21 = v55;
  LODWORD(v55) = 0;
  v57 = v21;
  v56 = v54;
  v44 = sub_10BBC70((__int64)&v56);
  sub_969240(&v56);
  sub_969240(&v54);
  v22 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
  v23 = *(_WORD *)v22;
  v53 = *(_QWORD *)(v22 + 8);
  LOWORD(v52) = v23;
  LODWORD(v54) = sub_327FC40(*(_QWORD **)(a10 + 64), v44);
  v55 = v24;
  if ( sub_32801E0((__int64)&v52) )
  {
    v51 = sub_3281590((__int64)&v52);
    LODWORD(v54) = sub_327FD70(*(__int64 **)(a10 + 64), v54, v55, v51);
    v55 = v37;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a10 + 16)
                                                                                             + 1752LL))(
          *(_QWORD *)(a10 + 16),
          229,
          (unsigned int)v52,
          v53,
          (unsigned int)v54,
          v55) )
    return 0;
  LOWORD(v25) = sub_3281100((unsigned __int16 *)&v54, 229);
  *(_QWORD *)&v27 = sub_33F7D60(a10, v25, v26);
  v28 = *(__int128 **)(a1 + 40);
  v45 = v27;
  sub_3285E70((__int64)&v56, a1);
  v29 = sub_3406EB0(a10, 229, (unsigned int)&v56, v54, v55, DWORD2(v45), *v28, v45);
  *(_QWORD *)&v45 = v30;
  sub_9C6650(&v56);
  v31 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v32 = *((_QWORD *)v31 + 1);
  v33 = *v31;
  sub_3285E70((__int64)&v56, a1);
  *(_QWORD *)&v45 = sub_33FB310(a10, v29, v45, &v56, v33, v32);
  sub_9C6650(&v56);
  return v45;
}
