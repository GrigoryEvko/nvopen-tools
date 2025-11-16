// Function: sub_26E9470
// Address: 0x26e9470
//
__int64 __fastcall sub_26E9470(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned int v5; // r14d
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  int v10; // r8d
  _BYTE *v11; // r8
  int v12; // edi
  __int64 i; // rdx
  unsigned __int64 v14; // rdx
  int v15; // eax
  unsigned int v16; // ecx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdx
  size_t v20; // r14
  unsigned int v21; // r14d
  unsigned __int64 v22; // rsi
  _BYTE *v23; // rdi
  int v24; // eax
  unsigned int v25; // edx
  unsigned int v26; // r9d
  unsigned int v27; // r8d
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // rcx
  size_t v31; // r14
  __int64 v32; // r14
  unsigned __int8 v33; // al
  unsigned __int8 **v34; // rdx
  unsigned __int8 *v35; // rax
  size_t v36; // rdx
  int *v37; // rsi
  unsigned __int8 *v38; // rcx
  unsigned __int8 v39; // al
  __int64 v40; // rdi
  __int64 v41; // rax
  unsigned __int8 v42; // al
  __int64 v43; // rdx
  __int64 v44; // r14
  __int64 v45; // r8
  __int64 v46; // [rsp+18h] [rbp-148h]
  __int64 v47; // [rsp+20h] [rbp-140h]
  int *v48; // [rsp+28h] [rbp-138h]
  unsigned __int8 *v49; // [rsp+28h] [rbp-138h]
  size_t v50; // [rsp+28h] [rbp-138h]
  __int64 v51; // [rsp+30h] [rbp-130h]
  int *v52; // [rsp+30h] [rbp-130h]
  __int64 v53; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v54; // [rsp+30h] [rbp-130h]
  __int64 v55; // [rsp+40h] [rbp-120h] BYREF
  __int64 v56; // [rsp+50h] [rbp-110h] BYREF
  __int64 v57; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v58; // [rsp+70h] [rbp-F0h] BYREF
  size_t v59; // [rsp+78h] [rbp-E8h]
  _QWORD v60[2]; // [rsp+80h] [rbp-E0h] BYREF
  int v61[52]; // [rsp+90h] [rbp-D0h] BYREF

  if ( !a1 )
    return 0;
  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) == 0 )
  {
    if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xF) != 2 )
      return 0;
    v3 = a1 - 16 - 8LL * ((v1 >> 2) & 0xF);
    goto LABEL_8;
  }
  if ( *(_DWORD *)(a1 - 24) == 2 )
  {
    v3 = *(_QWORD *)(a1 - 32);
LABEL_8:
    v4 = *(_QWORD *)(v3 + 8);
    if ( !v4 )
      return 0;
    v5 = *(_DWORD *)(v4 + 4);
    v46 = 0;
    if ( v5 <= 9 )
      goto LABEL_48;
    while ( v5 <= 0x63 )
    {
      v58 = v60;
      sub_2240A50((__int64 *)&v58, 2u, 0);
      v11 = v58;
LABEL_52:
      v44 = 2 * v5;
      v11[1] = a00010203040506[(unsigned int)(v44 + 1)];
      *v11 = a00010203040506[v44];
      while ( 2 )
      {
        v20 = v59;
        v52 = (int *)v58;
        sub_C7D030(v61);
        sub_C7D280(v61, v52, v20);
        sub_C7D290(v61, &v57);
        v53 = v57;
        if ( v58 != v60 )
          j_j___libc_free_0((unsigned __int64)v58);
        v21 = *(unsigned __int16 *)(v4 + 2);
        if ( v21 <= 9 )
        {
          v58 = v60;
          sub_2240A50((__int64 *)&v58, 1u, 0);
          v23 = v58;
LABEL_34:
          *v23 = v21 + 48;
          goto LABEL_35;
        }
        if ( v21 <= 0x63 )
        {
          v58 = v60;
          sub_2240A50((__int64 *)&v58, 2u, 0);
          v23 = v58;
          goto LABEL_54;
        }
        if ( v21 <= 0x3E7 )
        {
          v22 = 3;
LABEL_68:
          v58 = v60;
          goto LABEL_31;
        }
        if ( v21 <= 0x270F )
        {
          v22 = 4;
          goto LABEL_68;
        }
        v58 = v60;
        v22 = 5;
LABEL_31:
        sub_2240A50((__int64 *)&v58, v22, 0);
        v23 = v58;
        v24 = v59 - 1;
        do
        {
          v25 = v21;
          v26 = v21;
          v27 = 100 * (v21 / 0x64);
          v21 /= 0x64u;
          v28 = 2 * (v25 - v27);
          v29 = (unsigned int)(v28 + 1);
          LOBYTE(v28) = a00010203040506[v28];
          v23[v24] = a00010203040506[v29];
          v30 = (unsigned int)(v24 - 1);
          v24 -= 2;
          v23[v30] = v28;
        }
        while ( v26 > 0x270F );
        if ( v26 <= 0x3E7 )
          goto LABEL_34;
LABEL_54:
        v45 = 2 * v21;
        v23[1] = a00010203040506[(unsigned int)(v45 + 1)];
        *v23 = a00010203040506[v45];
LABEL_35:
        v31 = v59;
        v48 = (int *)v58;
        sub_C7D030(v61);
        sub_C7D280(v61, v48, v31);
        sub_C7D290(v61, &v56);
        v32 = v56 ^ v53;
        if ( v58 != v60 )
          j_j___libc_free_0((unsigned __int64)v58);
        v47 = v4 - 16;
        v33 = *(_BYTE *)(v4 - 16);
        if ( (v33 & 2) != 0 )
          v34 = *(unsigned __int8 ***)(v4 - 32);
        else
          v34 = (unsigned __int8 **)(v47 - 8LL * ((v33 >> 2) & 0xF));
        v35 = sub_AF34D0(*v34);
        v36 = 0;
        v37 = (int *)byte_3F871B3;
        v38 = v35;
        if ( v35 )
        {
          v49 = v35 - 16;
          v39 = *(v35 - 16);
          if ( (v39 & 2) != 0 )
          {
            v40 = *(_QWORD *)(*((_QWORD *)v38 - 4) + 24LL);
            if ( v40 )
              goto LABEL_42;
LABEL_64:
            v37 = *(int **)(*((_QWORD *)v38 - 4) + 16LL);
            if ( v37 )
            {
LABEL_59:
              v37 = (int *)sub_B91420((__int64)v37);
              goto LABEL_43;
            }
          }
          else
          {
            v40 = *(_QWORD *)&v49[-8 * ((v39 >> 2) & 0xF) + 24];
            if ( v40 )
            {
LABEL_42:
              v54 = v38;
              v41 = sub_B91420(v40);
              v38 = v54;
              v37 = (int *)v41;
              if ( v36 )
                goto LABEL_43;
              v39 = *(v54 - 16);
              if ( (v39 & 2) != 0 )
                goto LABEL_64;
            }
            v37 = *(int **)&v49[-8 * ((v39 >> 2) & 0xF) + 16];
            if ( v37 )
              goto LABEL_59;
          }
          v36 = 0;
        }
LABEL_43:
        v50 = v36;
        sub_C7D030(v61);
        sub_C7D280(v61, v37, v50);
        sub_C7D290(v61, &v55);
        v42 = *(_BYTE *)(v4 - 16);
        v46 ^= v55 ^ v32;
        if ( (v42 & 2) != 0 )
        {
          if ( *(_DWORD *)(v4 - 24) != 2 )
            return v46;
          v43 = *(_QWORD *)(v4 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
            return v46;
          v43 = v47 - 8LL * ((v42 >> 2) & 0xF);
        }
        v4 = *(_QWORD *)(v43 + 8);
        if ( !v4 )
          return v46;
        v5 = *(_DWORD *)(v4 + 4);
        if ( v5 <= 9 )
        {
LABEL_48:
          v58 = v60;
          sub_2240A50((__int64 *)&v58, 1u, 0);
          v11 = v58;
LABEL_23:
          *v11 = v5 + 48;
          continue;
        }
        break;
      }
    }
    if ( v5 <= 0x3E7 )
    {
      v8 = 3;
      v6 = v5;
    }
    else
    {
      v6 = v5;
      v7 = v5;
      if ( v5 > 0x270F )
      {
        LODWORD(v8) = 1;
        while ( 1 )
        {
          v9 = v7;
          v10 = v8;
          v8 = (unsigned int)(v8 + 4);
          v7 /= 0x2710u;
          if ( v9 <= 0x1869F )
            goto LABEL_18;
          if ( (unsigned int)v7 <= 0x63 )
            break;
          if ( (unsigned int)v7 <= 0x3E7 )
          {
            v8 = (unsigned int)(v10 + 6);
            goto LABEL_18;
          }
          if ( (unsigned int)v7 <= 0x270F )
          {
            v8 = (unsigned int)(v10 + 7);
            goto LABEL_18;
          }
        }
        v51 = v5;
        v8 = (unsigned int)(v10 + 5);
        v58 = v60;
LABEL_19:
        sub_2240A50((__int64 *)&v58, v8, 0);
        v11 = v58;
        v12 = v59 - 1;
        for ( i = v51; ; i = v5 )
        {
          v14 = (unsigned __int64)(1374389535 * i) >> 37;
          v15 = v5 - 100 * v14;
          v16 = v5;
          v5 = v14;
          v17 = (unsigned int)(2 * v15);
          v18 = (unsigned int)(v17 + 1);
          LOBYTE(v17) = a00010203040506[v17];
          v11[v12] = a00010203040506[v18];
          v19 = (unsigned int)(v12 - 1);
          v12 -= 2;
          v11[v19] = v17;
          if ( v16 <= 0x270F )
            break;
        }
        if ( v16 <= 0x3E7 )
          goto LABEL_23;
        goto LABEL_52;
      }
      v8 = 4;
    }
LABEL_18:
    v51 = v6;
    v58 = v60;
    goto LABEL_19;
  }
  return 0;
}
