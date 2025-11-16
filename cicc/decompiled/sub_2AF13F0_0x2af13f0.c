// Function: sub_2AF13F0
// Address: 0x2af13f0
//
void __fastcall sub_2AF13F0(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned int v3; // r15d
  unsigned __int64 v6; // rbx
  char *v7; // rax
  unsigned __int64 v8; // rdx
  unsigned int v9; // r13d
  unsigned int v10; // r14d
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  char v15; // dl
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // ebx
  unsigned int i; // ebx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // r8
  unsigned int v25; // eax
  unsigned __int8 *v26; // rbx
  unsigned int v27; // eax
  __int64 v28; // r9
  __int64 *v29; // r8
  unsigned __int8 *v30; // r13
  unsigned __int8 v31; // si
  __int64 v32; // r14
  int v33; // ecx
  int v34; // edx
  int v35; // r11d
  unsigned int v36; // ecx
  int *v37; // r9
  int v38; // edi
  unsigned int v39; // ecx
  __int64 v40; // r15
  int v41; // edx
  __int64 v42; // r9
  __int64 *v43; // r8
  __int64 v44; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v45; // [rsp+18h] [rbp-A8h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  unsigned __int64 v48; // [rsp+30h] [rbp-90h]
  unsigned __int64 v49; // [rsp+38h] [rbp-88h]
  unsigned __int64 v50; // [rsp+40h] [rbp-80h]
  unsigned __int64 v51; // [rsp+48h] [rbp-78h]
  unsigned __int8 *v52; // [rsp+50h] [rbp-70h] BYREF
  __int64 v53; // [rsp+58h] [rbp-68h]
  _BYTE v54[96]; // [rsp+60h] [rbp-60h] BYREF

  v3 = a2;
  v6 = HIDWORD(a2);
  v45 = (unsigned __int8 *)a2;
  sub_2AE5DB0(*(__int64 **)(a1 + 48));
  sub_2AB4590(*(_QWORD *)(a1 + 48), a2);
  v7 = sub_2AECC70(*(_QWORD *)(a1 + 48), a2, a3);
  v50 = (unsigned __int64)v7;
  v51 = v8;
  v9 = v8;
  if ( !((unsigned int)v7 | (unsigned int)v8) )
    return;
  v10 = (unsigned int)v7;
  v11 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(v11 + 108) && *(_DWORD *)(v11 + 100)
    || (a2 = **(_QWORD **)(*(_QWORD *)a1 + 32LL), (unsigned __int8)sub_31A6C30(*(_QWORD *)(v11 + 440), a2)) )
  {
    v44 = *(_QWORD *)(a1 + 32);
    v12 = sub_23DF0D0(dword_500E228);
    v15 = byte_500E2A8;
    if ( v12 <= 0 )
      v15 = sub_DFAE30(v44);
    if ( !v15 && (unsigned __int8)sub_2AAEF20(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 504LL), a2) )
      sub_2AC31B0(*(_QWORD *)(a1 + 48));
  }
  v16 = *(_QWORD *)(a1 + 48);
  if ( !*(_BYTE *)(v16 + 108) || !*(_DWORD *)(v16 + 100) )
  {
    if ( (_BYTE)v6 )
      goto LABEL_10;
LABEL_34:
    v27 = v10;
    if ( !v3 )
      goto LABEL_11;
    goto LABEL_35;
  }
  sub_31AA3D0(*(_QWORD *)(a1 + 40));
  if ( !(_BYTE)v6 )
    goto LABEL_34;
LABEL_10:
  if ( v3 )
  {
    if ( !BYTE4(v51) )
    {
LABEL_36:
      v28 = *(_QWORD *)a1;
      v29 = *(__int64 **)(a1 + 80);
      v52 = 0;
      sub_2AB8CE0(
        "UserVF ignored because it may be larger than the maximal safe VF",
        0x40u,
        (__int64)"InvalidUserVF",
        13,
        v29,
        v28,
        &v52);
      sub_9C6650(&v52);
      goto LABEL_11;
    }
    v27 = v9;
LABEL_35:
    if ( v3 <= v27 )
    {
      sub_2AD3790(*(_QWORD *)(a1 + 48));
      v40 = *(_QWORD *)(a1 + 48);
      v52 = v45;
      sub_2AE5AE0(v40, (unsigned __int64)v45);
      sub_2AD3030(v40, (__int64)v45);
      sub_2AD1E10(v40, (__int64)v45);
      if ( !v41 )
      {
        sub_2AF1190(a1, (unsigned __int64)v45, (unsigned __int64)v45);
        return;
      }
      v42 = *(_QWORD *)a1;
      v43 = *(__int64 **)(a1 + 80);
      v52 = 0;
      sub_2AB8CE0("UserVF ignored because of invalid costs.", 0x28u, (__int64)"InvalidCost", 11, v43, v42, &v52);
      sub_9C6650(&v52);
      goto LABEL_11;
    }
    goto LABEL_36;
  }
LABEL_11:
  v52 = v54;
  v53 = 0x600000000LL;
  if ( v10 )
  {
    v17 = 6;
    v18 = 0;
    v19 = 1;
    while ( 1 )
    {
      v13 = v18 + 1;
      LODWORD(v46) = v19;
      BYTE4(v46) = 0;
      if ( v18 + 1 > v17 )
      {
        sub_C8D5F0((__int64)&v52, v54, v18 + 1, 8u, v13, v14);
        v18 = (unsigned int)v53;
      }
      v19 *= 2;
      *(_QWORD *)&v52[8 * v18] = v46;
      v18 = (unsigned int)(v53 + 1);
      LODWORD(v53) = v53 + 1;
      if ( v10 < v19 )
        break;
      v17 = HIDWORD(v53);
    }
  }
  if ( BYTE4(v51) )
  {
    for ( i = 1; v9 >= i; i *= 2 )
    {
      v21 = (unsigned int)v53;
      LODWORD(v47) = i;
      BYTE4(v47) = 1;
      v22 = (unsigned int)v53 + 1LL;
      if ( v22 > HIDWORD(v53) )
      {
        sub_C8D5F0((__int64)&v52, v54, v22, 8u, v13, v14);
        v21 = (unsigned int)v53;
      }
      *(_QWORD *)&v52[8 * v21] = v47;
      LODWORD(v53) = v53 + 1;
    }
  }
  sub_2AD3790(*(_QWORD *)(a1 + 48));
  v26 = v52;
  v30 = &v52[8 * (unsigned int)v53];
  if ( v30 != v52 )
  {
    while ( 1 )
    {
      v31 = v26[4];
      v32 = *(_QWORD *)(a1 + 48);
      v49 = *(_QWORD *)v26;
      v25 = *(_DWORD *)v26;
      if ( v31 )
      {
        v23 = *(_DWORD *)(v32 + 184);
        v24 = *(_QWORD *)(v32 + 168);
        if ( v23 )
        {
          v33 = 37 * v25 - 1;
LABEL_42:
          v34 = v23 - 1;
          v35 = 1;
          v36 = v34 & v33;
          while ( 2 )
          {
            v37 = (int *)(v24 + 72LL * v36);
            v38 = *v37;
            if ( v25 == *v37 )
            {
              if ( v31 == *((_BYTE *)v37 + 4) )
                goto LABEL_27;
              if ( v38 != -1 )
                goto LABEL_45;
            }
            else if ( v38 != -1 )
            {
LABEL_45:
              v39 = v35 + v36;
              ++v35;
              v36 = v34 & v39;
              continue;
            }
            break;
          }
          if ( *((_BYTE *)v37 + 4) )
            goto LABEL_26;
          goto LABEL_45;
        }
      }
      else
      {
        if ( v25 == 1 )
          goto LABEL_30;
        v23 = *(_DWORD *)(v32 + 184);
        v24 = *(_QWORD *)(v32 + 168);
        if ( v23 )
        {
          v33 = 37 * v25;
          goto LABEL_42;
        }
      }
LABEL_26:
      sub_2ACAC50(v32, v49);
      sub_2AE4570(v32, v49);
      sub_2AC7F80(v32, v49);
      sub_2ADE2D0(v32, v49);
      v25 = *(_DWORD *)v26;
LABEL_27:
      if ( !v26[4] )
        break;
      if ( v25 )
        goto LABEL_29;
LABEL_30:
      v26 += 8;
      if ( v30 == v26 )
        goto LABEL_31;
    }
    if ( v25 <= 1 )
      goto LABEL_30;
LABEL_29:
    sub_2AD3030(*(_QWORD *)(a1 + 48), *(_QWORD *)v26);
    goto LABEL_30;
  }
LABEL_31:
  BYTE4(v48) = 0;
  LODWORD(v48) = 1;
  sub_2AF1190(a1, v48, v50);
  BYTE4(v49) = 1;
  LODWORD(v49) = 1;
  sub_2AF1190(a1, v49, v51);
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
}
