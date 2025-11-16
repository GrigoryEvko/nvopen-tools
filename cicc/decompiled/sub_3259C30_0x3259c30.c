// Function: sub_3259C30
// Address: 0x3259c30
//
__int64 __fastcall sub_3259C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // r12
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 i; // r8
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  unsigned int v14; // r11d
  __int64 *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  int v18; // r9d
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // r8
  unsigned int v22; // r9d
  unsigned __int64 v23; // r14
  __int64 v24; // rdx
  unsigned __int64 *v25; // rdx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // r8
  unsigned __int64 v32; // r10
  unsigned __int64 v33; // r9
  __int64 v34; // rax
  unsigned __int64 *v35; // rax
  int v36; // edx
  int v37; // r10d
  __int64 v39; // [rsp+10h] [rbp-110h]
  unsigned __int64 v40; // [rsp+18h] [rbp-108h]
  __int64 v42; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v43; // [rsp+30h] [rbp-F0h]
  __int64 v44; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v45; // [rsp+38h] [rbp-E8h]
  __int64 v46; // [rsp+38h] [rbp-E8h]
  int v47; // [rsp+40h] [rbp-E0h]
  __int64 v48; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v49; // [rsp+40h] [rbp-E0h]
  __int64 v50; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v51; // [rsp+40h] [rbp-E0h]
  __int64 v52; // [rsp+48h] [rbp-D8h]
  unsigned int v53; // [rsp+48h] [rbp-D8h]
  unsigned int v54; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v55; // [rsp+48h] [rbp-D8h]
  __int64 v56; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+58h] [rbp-C8h]
  __int64 v58; // [rsp+60h] [rbp-C0h]
  __int64 v59; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v60; // [rsp+70h] [rbp-B0h]
  __int64 v61; // [rsp+78h] [rbp-A8h]
  __int64 v62; // [rsp+80h] [rbp-A0h]
  unsigned int v63; // [rsp+88h] [rbp-98h]
  char v64; // [rsp+90h] [rbp-90h]
  unsigned int v65; // [rsp+94h] [rbp-8Ch]
  __int64 v66; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-78h]
  __int64 v68; // [rsp+B0h] [rbp-70h]
  __int64 v69; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v70; // [rsp+C0h] [rbp-60h]
  __int64 v71; // [rsp+C8h] [rbp-58h]
  __int64 v72; // [rsp+D0h] [rbp-50h]
  unsigned int v73; // [rsp+D8h] [rbp-48h]
  char v74; // [rsp+E0h] [rbp-40h]
  unsigned int v75; // [rsp+E4h] [rbp-3Ch]

  result = a2 + 320;
  v6 = *(_QWORD *)(a2 + 328);
  v42 = result;
  if ( v6 == result )
    return result;
  for ( i = v6; ; i = v10 )
  {
    result = v42;
    v10 = i;
    do
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( v10 == v42 )
      {
        if ( *(_BYTE *)(i + 236) )
          return result;
        if ( *(_QWORD *)(a2 + 328) == i )
          goto LABEL_32;
LABEL_8:
        v52 = i;
        v11 = sub_AA4FF0(*(_QWORD *)(i + 16));
        v12 = *(_QWORD *)(a3 + 40);
        v13 = *(_DWORD *)(a3 + 56);
        if ( v11 )
          v11 -= 24;
        if ( v13 )
        {
          v14 = (v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v11 == *v15 )
          {
LABEL_12:
            v47 = *((_DWORD *)v15 + 2);
            v17 = sub_3258B50(v52);
            v18 = v47;
            i = v52;
            v19 = v17;
            goto LABEL_13;
          }
          v36 = 1;
          while ( v16 != -4096 )
          {
            v37 = v36 + 1;
            v14 = (v13 - 1) & (v36 + v14);
            v15 = (__int64 *)(v12 + 16LL * v14);
            v16 = *v15;
            if ( v11 == *v15 )
              goto LABEL_12;
            v36 = v37;
          }
        }
        v15 = (__int64 *)(v12 + 16LL * v13);
        goto LABEL_12;
      }
    }
    while ( !*(_BYTE *)(v10 + 235) );
    if ( !*(_BYTE *)(i + 236) )
      break;
LABEL_28:
    ;
  }
  if ( *(_QWORD *)(a2 + 328) != i )
    goto LABEL_8;
LABEL_32:
  v18 = -1;
  v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 536LL);
LABEL_13:
  v48 = i;
  v53 = v18;
  v20 = sub_3258F50(a1, v19);
  v21 = v48;
  v22 = v53;
  v23 = v40 & 0xFFFFFFFF00000000LL | v53;
  v24 = *(unsigned int *)(a4 + 8);
  v40 = v23;
  if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    v46 = v48;
    v51 = v20;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v24 + 1, 0x10u, v21, v53);
    v24 = *(unsigned int *)(a4 + 8);
    v21 = v46;
    v20 = v51;
    v22 = v53;
  }
  v25 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v24);
  v44 = v21;
  v25[1] = v23;
  *v25 = v20;
  ++*(_DWORD *)(a4 + 8);
  v26 = *(_QWORD *)(v21 + 56);
  v54 = v22;
  v27 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
  v66 = a3;
  v49 = v26;
  v70 = v27 + 48;
  v68 = v10;
  v69 = v10;
  v75 = v22;
  v73 = v22;
  v67 = 0;
  v74 = 0;
  v71 = 0;
  v72 = 0;
  sub_32588C0((__int64)&v66);
  v59 = v10;
  v56 = a3;
  v60 = v49;
  v57 = 0;
  v58 = v44;
  v64 = 0;
  v65 = v54;
  v61 = 0;
  v62 = 0;
  v63 = v54;
  sub_32588C0((__int64)&v56);
  v39 = v10;
  v28 = v58;
  v66 = v56;
  v50 = v67;
  v67 = v57;
  v55 = v70;
  v29 = v68;
  v69 = v59;
  v68 = v58;
  v70 = v60;
  v71 = v61;
  v72 = v62;
  v73 = v63;
  v74 = v64;
  v75 = v65;
  while ( 1 )
  {
    if ( v29 == v28 && v70 == v55 )
    {
      result = v50;
      if ( v67 == v50 )
        break;
    }
    v30 = v72;
    if ( !v72 )
      v30 = v71;
    if ( *(_BYTE *)(a1 + 28) || *(_BYTE *)(a1 + 29) )
      v32 = sub_E808D0(v30, 0x72u, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0);
    else
      v32 = sub_3258F90(a1, v30);
    v33 = v73 | v4 & 0xFFFFFFFF00000000LL;
    v34 = *(unsigned int *)(a4 + 8);
    v4 = v33;
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v43 = v33;
      v45 = v32;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v34 + 1, 0x10u, v31, v33);
      v34 = *(unsigned int *)(a4 + 8);
      v33 = v43;
      v32 = v45;
    }
    v35 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v34);
    *v35 = v32;
    v35[1] = v33;
    ++*(_DWORD *)(a4 + 8);
    sub_32588C0((__int64)&v66);
    v28 = v68;
  }
  v10 = v39;
  if ( v39 != v42 )
    goto LABEL_28;
  return result;
}
