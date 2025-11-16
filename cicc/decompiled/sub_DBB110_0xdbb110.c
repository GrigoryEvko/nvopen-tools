// Function: sub_DBB110
// Address: 0xdbb110
//
__int64 __fastcall sub_DBB110(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v6; // ebx
  __int64 v7; // r15
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // edi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned int v27; // eax
  unsigned int v28; // eax
  int v29; // edx
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rax
  unsigned int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // [rsp+20h] [rbp-150h]
  unsigned int v36; // [rsp+20h] [rbp-150h]
  unsigned __int64 v37; // [rsp+28h] [rbp-148h]
  bool v38; // [rsp+37h] [rbp-139h] BYREF
  unsigned __int8 *v39; // [rsp+38h] [rbp-138h] BYREF
  __int64 v40; // [rsp+40h] [rbp-130h] BYREF
  __int64 v41; // [rsp+48h] [rbp-128h] BYREF
  __int64 v42[2]; // [rsp+50h] [rbp-120h] BYREF
  __int64 v43[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v44[2]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v45[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v46; // [rsp+90h] [rbp-E0h] BYREF
  int v47; // [rsp+98h] [rbp-D8h]
  const void *v48; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v49; // [rsp+A8h] [rbp-C8h]
  const void *v50; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v51; // [rsp+B8h] [rbp-B8h]
  __int64 v52; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v53; // [rsp+C8h] [rbp-A8h]
  __int64 v54; // [rsp+D0h] [rbp-A0h] BYREF
  int v55; // [rsp+D8h] [rbp-98h]
  __int64 v56[2]; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v57; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v58[2]; // [rsp+100h] [rbp-70h] BYREF
  __int64 v59[2]; // [rsp+110h] [rbp-60h] BYREF
  __int64 v60; // [rsp+120h] [rbp-50h] BYREF
  int v61; // [rsp+128h] [rbp-48h]
  __int64 v62[8]; // [rsp+130h] [rbp-40h] BYREF

  v37 = a2[1];
  v6 = sub_D97050((__int64)a2, *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8LL));
  sub_AADB10((__int64)&v48, v6, 1);
  v7 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)v7 != 84 )
    goto LABEL_2;
  v12 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 16LL);
  if ( v12 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v12 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v13 - 30) <= 0xAu )
        break;
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        goto LABEL_23;
    }
    v14 = a2[5];
    v15 = *(_DWORD *)(v14 + 32);
LABEL_19:
    v16 = *(_QWORD *)(v13 + 40);
    if ( v16 )
    {
      v17 = (unsigned int)(*(_DWORD *)(v16 + 44) + 1);
      if ( *(_DWORD *)(v16 + 44) + 1 >= v15 )
        goto LABEL_35;
    }
    else
    {
      v17 = 0;
      if ( !v15 )
        goto LABEL_35;
    }
    if ( *(_QWORD *)(*(_QWORD *)(v14 + 24) + 8 * v17) )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
          goto LABEL_23;
        v13 = *(_QWORD *)(v12 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v13 - 30) <= 0xAu )
          goto LABEL_19;
      }
    }
LABEL_35:
    v27 = v49;
    *(_DWORD *)(a1 + 8) = v49;
    if ( v27 > 0x40 )
      sub_C43780(a1, &v48);
    else
      *(_QWORD *)a1 = v48;
    v11 = v51;
    *(_DWORD *)(a1 + 24) = v51;
    if ( v11 <= 0x40 )
    {
      *(_QWORD *)(a1 + 16) = v50;
      goto LABEL_11;
    }
    goto LABEL_10;
  }
LABEL_23:
  if ( (unsigned __int8)sub_990E50(v7, &v39, &v40, &v41) )
  {
    v19 = a2[6];
    v20 = *(_QWORD *)(v7 + 40);
    v21 = *(unsigned int *)(v19 + 24);
    v22 = *(_QWORD *)(v19 + 8);
    if ( (_DWORD)v21 )
    {
      v21 = (unsigned int)(v21 - 1);
      v18 = (unsigned int)v21 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v23 = (__int64 *)(v22 + 16 * v18);
      v24 = *v23;
      if ( *v23 == v20 )
      {
LABEL_26:
        v25 = v23[1];
LABEL_27:
        v35 = v25;
        if ( !(unsigned __int8)sub_B19060(v25 + 56, *((_QWORD *)v39 + 5), v18, v21)
          || (unsigned int)*v39 - 54 > 2
          || (v26 = *((_QWORD *)v39 - 8), v7 != v26)
          || !v26
          || (v28 = sub_DBB070((__int64)a2, v35, 0)) == 0
          || v28 >= v6 )
        {
          sub_AAF450(a1, (__int64)&v48);
          v11 = v51;
          goto LABEL_11;
        }
        v36 = v28;
        sub_9AC3E0((__int64)&v52, v40, v37, 0, a2[4], 0, a2[5], 1);
        sub_9AC3E0((__int64)v56, v41, v37, 0, a2[4], 0, a2[5], 1);
        sub_D95160((__int64)v42, (__int64)v56);
        sub_9691E0((__int64)v43, v6, v36 - 1, 0, 0);
        v38 = 0;
        sub_C49BE0((__int64)v44, (__int64)v42, (__int64)v43, &v38);
        if ( !v38 )
        {
          v29 = *v39;
          switch ( v29 )
          {
            case '7':
              sub_987BA0((__int64)&v60, v44);
              sub_C75B70(v58, (__int64)&v52, (__int64)&v60, 0, 0);
              sub_969240(v62);
              sub_969240(&v60);
              sub_D95160((__int64)&v46, (__int64)&v52);
              sub_C46A40((__int64)&v46, 1);
              v61 = v47;
              v60 = v46;
              v47 = 0;
              sub_9865C0((__int64)v45, (__int64)v59);
              sub_9875E0(a1, v45, &v60);
              sub_969240(v45);
              sub_969240(&v60);
              sub_969240(&v46);
              sub_969240(v59);
              sub_969240(v58);
              goto LABEL_50;
            case '8':
              sub_987BA0((__int64)&v60, v44);
              sub_C76560(v58, (__int64)&v52, (__int64)&v60, 0, 0);
              sub_969240(v62);
              sub_969240(&v60);
              if ( sub_986C60(&v52, v53 - 1) )
              {
                sub_D95160((__int64)&v46, (__int64)&v52);
                sub_C46A40((__int64)&v46, 1);
                v61 = v47;
                v47 = 0;
                v60 = v46;
                sub_9865C0((__int64)v45, (__int64)v59);
                sub_9875E0(a1, v45, &v60);
                sub_969240(v45);
                sub_969240(&v60);
                sub_969240(&v46);
                goto LABEL_49;
              }
              if ( sub_986C60(&v54, v55 - 1) )
              {
                sub_D95160((__int64)&v46, (__int64)v58);
                sub_C46A40((__int64)&v46, 1);
                v61 = v47;
                v47 = 0;
                v60 = v46;
                sub_9865C0((__int64)v45, (__int64)&v54);
                sub_9875E0(a1, v45, &v60);
                sub_969240(v45);
                sub_969240(&v60);
                sub_969240(&v46);
LABEL_49:
                sub_969240(v59);
                sub_969240(v58);
LABEL_50:
                sub_969240(v44);
                sub_969240(v43);
                sub_969240(v42);
                sub_969240(&v57);
                sub_969240(v56);
                sub_969240(&v54);
                sub_969240(&v52);
                v11 = v51;
                goto LABEL_11;
              }
              break;
            case '6':
              sub_987BA0((__int64)&v60, v44);
              sub_C74E10((__int64)v58, (__int64)&v52, (__int64)&v60, 0, 0, 0);
              sub_969240(v62);
              sub_969240(&v60);
              v33 = v53;
              if ( v53 > 0x40 )
              {
                v33 = sub_C44500((__int64)&v52);
              }
              else if ( v53 )
              {
                _BitScanReverse64(&v34, ~(v52 << (64 - (unsigned __int8)v53)));
                v33 = v34 ^ 0x3F;
                if ( v52 << (64 - (unsigned __int8)v53) == -1 )
                  v33 = 64;
              }
              if ( sub_986EE0((__int64)v44, v33) )
              {
                sub_D95160((__int64)&v46, (__int64)v58);
                sub_C46A40((__int64)&v46, 1);
                v61 = v47;
                v47 = 0;
                v60 = v46;
                sub_9865C0((__int64)v45, (__int64)&v54);
                sub_AADC30(a1, (__int64)v45, &v60);
                sub_969240(v45);
                sub_969240(&v60);
                sub_969240(&v46);
                sub_969240(v59);
                sub_969240(v58);
                goto LABEL_50;
              }
              break;
            default:
              BUG();
          }
          sub_969240(v59);
          sub_969240(v58);
        }
        sub_AAF450(a1, (__int64)&v48);
        goto LABEL_50;
      }
      v30 = 1;
      while ( v24 != -4096 )
      {
        v31 = v30 + 1;
        v32 = (unsigned int)v21 & ((_DWORD)v18 + v30);
        v18 = (unsigned int)v32;
        v23 = (__int64 *)(v22 + 16 * v32);
        v24 = *v23;
        if ( v20 == *v23 )
          goto LABEL_26;
        v30 = v31;
      }
    }
    v25 = 0;
    goto LABEL_27;
  }
LABEL_2:
  v8 = v49;
  *(_DWORD *)(a1 + 8) = v49;
  if ( v8 > 0x40 )
    sub_C43780(a1, &v48);
  else
    *(_QWORD *)a1 = v48;
  v9 = v51;
  *(_DWORD *)(a1 + 24) = v51;
  if ( v9 <= 0x40 )
  {
    *(_QWORD *)(a1 + 16) = v50;
    goto LABEL_6;
  }
LABEL_10:
  sub_C43780(a1 + 16, &v50);
  v11 = v51;
LABEL_11:
  if ( v11 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
LABEL_6:
  if ( v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  return a1;
}
