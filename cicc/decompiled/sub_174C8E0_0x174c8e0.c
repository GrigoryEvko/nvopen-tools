// Function: sub_174C8E0
// Address: 0x174c8e0
//
__int64 __fastcall sub_174C8E0(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 *v15; // rcx
  __int64 v16; // rdi
  unsigned __int8 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 **v25; // rax
  __int64 v26; // rdi
  unsigned __int8 *v27; // rax
  int v28; // r8d
  int v29; // r9d
  int v30; // edx
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 *v33; // r13
  __int64 v34; // rdx
  unsigned __int64 v35; // r15
  __int64 *v36; // rax
  int v37; // ecx
  const char *v38; // rax
  __int64 v39; // r14
  __int64 *v40; // r13
  __int64 v41; // rdx
  __int64 v42; // rax
  int v43; // r8d
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rax
  int v47; // r8d
  __int64 *v48; // r10
  __int64 *v49; // rcx
  __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 *v52; // rax
  bool v53; // al
  __int64 *v54; // rax
  int v55; // [rsp+0h] [rbp-D0h]
  int v56; // [rsp+4h] [rbp-CCh]
  unsigned int v57; // [rsp+8h] [rbp-C8h]
  __int64 v58; // [rsp+8h] [rbp-C8h]
  __int64 v59; // [rsp+18h] [rbp-B8h]
  _QWORD v60[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD *v61; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v62; // [rsp+40h] [rbp-90h]
  __int64 *v63; // [rsp+50h] [rbp-80h] BYREF
  __int64 v64; // [rsp+58h] [rbp-78h]
  _WORD v65[56]; // [rsp+60h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a2 - 24);
  v12 = *(_QWORD *)v11;
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
    v12 = **(_QWORD **)(v12 + 16);
  v13 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v13 = **(_QWORD **)(v13 + 16);
  v14 = *(__int64 **)(v12 + 24);
  if ( *(__int64 **)(v13 + 24) != v14 )
  {
    v15 = (__int64 *)sub_1646BA0(v14, *(_DWORD *)(v13 + 8) >> 8);
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v15 = sub_16463B0(v15, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
    v16 = a1[1];
    v65[0] = 257;
    v17 = sub_1708970(v16, 48, v11, (__int64 **)v15, (__int64 *)&v63);
    v18 = *(_QWORD *)a2;
    v65[0] = 257;
    v19 = (__int64)v17;
    v20 = sub_1648A60(56, 1u);
    v21 = v20;
    if ( v20 )
      sub_15FD590((__int64)v20, v19, v18, (__int64)&v63, 0);
    return (__int64)v21;
  }
  if ( *(_BYTE *)(v11 + 16) == 56 )
  {
    v23 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
    v24 = *(_QWORD *)v23;
    if ( *(_BYTE *)(*(_QWORD *)v23 + 8LL) == 16 )
      v24 = **(_QWORD **)(v24 + 16);
    v25 = (__int64 **)sub_1646BA0(*(__int64 **)(v24 + 24), *(_DWORD *)(v13 + 8) >> 8);
    v26 = a1[1];
    v65[0] = 257;
    v27 = sub_1708970(v26, 48, v23, v25, (__int64 *)&v63);
    v30 = *(_DWORD *)(v11 + 20);
    v31 = (__int64)v27;
    v63 = (__int64 *)v65;
    v32 = 24 * (1LL - (v30 & 0xFFFFFFF));
    v64 = 0x800000000LL;
    v33 = (__int64 *)(v11 + v32);
    v34 = -v32;
    v35 = 0xAAAAAAAAAAAAAAABLL * (v34 >> 3);
    v36 = (__int64 *)v65;
    v37 = 0;
    if ( (unsigned __int64)v34 > 0xC0 )
    {
      sub_16CD150((__int64)&v63, v65, 0xAAAAAAAAAAAAAAABLL * (v34 >> 3), 8, v28, v29);
      v37 = v64;
      v36 = &v63[(unsigned int)v64];
    }
    if ( (__int64 *)v11 != v33 )
    {
      do
      {
        if ( v36 )
          *v36 = *v33;
        v33 += 3;
        ++v36;
      }
      while ( (__int64 *)v11 != v33 );
      v37 = v64;
    }
    LODWORD(v64) = v37 + v35;
    v38 = sub_1649960(v11);
    v39 = (unsigned int)v64;
    v40 = v63;
    v60[0] = v38;
    v62 = 261;
    v60[1] = v41;
    v61 = v60;
    v42 = *(_QWORD *)v31;
    if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) == 16 )
      v42 = **(_QWORD **)(v42 + 16);
    v57 = v64 + 1;
    v59 = *(_QWORD *)(v42 + 24);
    v21 = sub_1648A60(72, (int)v64 + 1);
    if ( v21 )
    {
      v43 = v57;
      v44 = *(_QWORD *)v31;
      v58 = (__int64)&v21[-3 * v57];
      if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) == 16 )
        v44 = **(_QWORD **)(v44 + 16);
      v55 = v43;
      v56 = *(_DWORD *)(v44 + 8) >> 8;
      v45 = (__int64 *)sub_15F9F50(v59, (__int64)v40, v39);
      v46 = (__int64 *)sub_1646BA0(v45, v56);
      v47 = v55;
      v48 = v46;
      if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) == 16 )
      {
        v54 = sub_16463B0(v46, *(_QWORD *)(*(_QWORD *)v31 + 32LL));
        v47 = v55;
        v48 = v54;
      }
      else
      {
        v49 = &v40[v39];
        if ( v40 != v49 )
        {
          v50 = v40;
          while ( 1 )
          {
            v51 = *(_QWORD *)*v50;
            if ( *(_BYTE *)(v51 + 8) == 16 )
              break;
            if ( v49 == ++v50 )
              goto LABEL_33;
          }
          v52 = sub_16463B0(v48, *(_QWORD *)(v51 + 32));
          v47 = v55;
          v48 = v52;
        }
      }
LABEL_33:
      sub_15F1EA0((__int64)v21, (__int64)v48, 32, v58, v47, 0);
      v21[7] = v59;
      v21[8] = sub_15F9F50(v59, (__int64)v40, v39);
      sub_15F9CE0((__int64)v21, v31, v40, v39, (__int64)&v61);
    }
    v53 = sub_15FA300(v11);
    sub_15FA2E0((__int64)v21, v53);
    if ( v63 != (__int64 *)v65 )
      _libc_free((unsigned __int64)v63);
    return (__int64)v21;
  }
  return sub_174C560(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
