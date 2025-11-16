// Function: sub_28D2A80
// Address: 0x28d2a80
//
__int64 __fastcall sub_28D2A80(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r13
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rsi
  _QWORD *v14; // rax
  char *v15; // rcx
  _QWORD **v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned __int8 *v20; // r9
  __int64 *v21; // rax
  __int64 *i; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int8 **v27; // rbx
  char v28; // r11
  __int64 v29; // r8
  unsigned __int8 **v30; // rdi
  unsigned __int8 **v31; // rax
  unsigned __int8 v32; // r10
  unsigned __int8 *v33; // rax
  unsigned __int8 **v35; // rsi
  int v36; // edx
  unsigned __int8 **v37; // rdx
  int v38; // eax
  int v39; // ecx
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  unsigned __int8 **v43; // rax
  unsigned __int8 *v44; // rdi
  unsigned __int8 *v45; // rbx
  _BYTE *v46; // rsi
  char v47; // al
  __int64 v48; // rsi
  char v49; // al
  __int64 v50; // rcx
  __int64 v51; // r14
  __int64 *v52; // r13
  __int64 *v53; // rcx
  char v54; // al
  unsigned int v55; // eax
  unsigned int v56; // r10d
  unsigned __int8 *v57; // rax
  int j; // eax
  int v59; // r8d
  __int64 *v61; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v63; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v64; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v65; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v67; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v68; // [rsp+18h] [rbp-D8h]
  char v69; // [rsp+2Eh] [rbp-C2h] BYREF
  char v70; // [rsp+2Fh] [rbp-C1h] BYREF
  __int64 v71; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v72; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD v73[8]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 *v74; // [rsp+80h] [rbp-70h] BYREF
  __int64 v75; // [rsp+88h] [rbp-68h]
  __int64 *v76; // [rsp+90h] [rbp-60h]
  __int64 v77; // [rsp+98h] [rbp-58h]
  __int64 *v78; // [rsp+A0h] [rbp-50h]
  char *v79; // [rsp+A8h] [rbp-48h]
  char *v80; // [rsp+B0h] [rbp-40h]

  v7 = (unsigned int)a3;
  v72 = a5;
  v69 = 0;
  v70 = 1;
  v71 = a4;
  v9 = sub_A777F0(0x38u, (__int64 *)(a1 + 72));
  if ( v9 )
  {
    v10 = v72;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 8) = 0xFFFFFFFD00000008LL;
    *(_DWORD *)(v9 + 32) = a3;
    *(_QWORD *)(v9 + 24) = 0;
    *(_DWORD *)(v9 + 36) = 0;
    *(_QWORD *)(v9 + 40) = 0;
    *(_QWORD *)v9 = &unk_4A21B70;
    *(_QWORD *)(v9 + 48) = v10;
  }
  else
  {
    v7 = MEMORY[0x20];
  }
  if ( v7 && (--v7, v7) )
  {
    _BitScanReverse64(&v7, v7);
    v11 = 64 - (v7 ^ 0x3F);
    v7 = (int)v11;
    if ( v11 >= *(_DWORD *)(a1 + 176) )
      goto LABEL_6;
  }
  else
  {
    LOBYTE(v11) = 0;
    if ( !*(_DWORD *)(a1 + 176) )
      goto LABEL_6;
  }
  v16 = (_QWORD **)(*(_QWORD *)(a1 + 168) + 8 * v7);
  v14 = *v16;
  if ( *v16 )
  {
    *v16 = (_QWORD *)*v14;
    goto LABEL_12;
  }
LABEL_6:
  v12 = *(_QWORD *)(a1 + 72);
  v13 = 8LL << v11;
  *(_QWORD *)(a1 + 152) += 8LL << v11;
  v14 = (_QWORD *)((v12 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v15 = (char *)v14 + (8LL << v11);
  if ( *(_QWORD *)(a1 + 80) >= (unsigned __int64)v15 && v12 )
    *(_QWORD *)(a1 + 72) = v15;
  else
    v14 = (_QWORD *)sub_9D1E70(a1 + 72, v13, v13, 3);
LABEL_12:
  *(_QWORD *)(v9 + 24) = v14;
  v17 = (__int64)&a2[2 * a3];
  v18 = *(_QWORD *)(*a2 + 8);
  *(_DWORD *)(v9 + 12) = 55;
  v73[0] = v17;
  *(_QWORD *)(v9 + 40) = v18;
  v73[1] = v17;
  v73[2] = &v71;
  v73[4] = &v72;
  v73[5] = &v70;
  v73[6] = &v69;
  v73[3] = a1;
  sub_28CB230((__int64)v73);
  v80 = &v69;
  v75 = v17;
  v79 = &v70;
  v74 = a2;
  v76 = &v71;
  v77 = a1;
  v78 = &v72;
  sub_28CB230((__int64)&v74);
  v21 = v74;
  for ( i = (__int64 *)v73[0]; v74 != i; v21 = v74 )
  {
    v23 = sub_28C86C0(a1, *v21);
    v24 = *(_QWORD *)(v9 + 24);
    v25 = v23;
    v26 = *(unsigned int *)(v9 + 36);
    *(_DWORD *)(v9 + 36) = v26 + 1;
    *(_QWORD *)(v24 + 8 * v26) = v25;
    v74 += 2;
    sub_28CB230((__int64)&v74);
  }
  v27 = *(unsigned __int8 ***)(v9 + 24);
  v28 = 0;
  v29 = 0;
  v30 = &v27[*(unsigned int *)(v9 + 36)];
  v31 = v27;
  if ( v27 == v30 )
    goto LABEL_62;
  while ( 1 )
  {
    v20 = *v31;
    v19 = (__int64)v31;
    v32 = **v31;
    if ( v32 == 13 )
    {
      v28 = 1;
      goto LABEL_18;
    }
    if ( (unsigned int)v32 - 12 > 1 )
      break;
    v29 = 1;
LABEL_18:
    if ( v30 == ++v31 )
    {
      if ( (_BYTE)v29 )
      {
        v57 = (unsigned __int8 *)sub_ACA8A0(*(__int64 ***)(a4 + 8));
        return sub_28CECC0(a1, v57);
      }
      if ( v28 )
      {
        v33 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(a4 + 8));
        return sub_28CECC0(a1, v33);
      }
LABEL_62:
      sub_28C79C0(*(_QWORD **)(v9 + 24), *(_DWORD *)(v9 + 32), a1 + 168, v19, v29, (__int64)v20);
      return *(_QWORD *)(a1 + 2048);
    }
  }
  v35 = v31 + 1;
  if ( v30 != v31 + 1 )
  {
    do
    {
      v36 = **v35;
      if ( (_BYTE)v36 == 13 )
      {
        v28 = 1;
      }
      else
      {
        if ( (unsigned int)(v36 - 12) > 1 )
          break;
        v29 = 1;
      }
      ++v35;
    }
    while ( v30 != v35 );
    if ( v30 != v31 )
    {
LABEL_29:
      v37 = (unsigned __int8 **)(v19 + 8);
      if ( v30 == (unsigned __int8 **)(v19 + 8) )
        goto LABEL_34;
      do
      {
        v19 = (__int64)v37;
        v38 = **v37;
        if ( (_BYTE)v38 == 13 )
        {
          v28 = 1;
        }
        else
        {
          if ( (unsigned int)(v38 - 12) > 1 )
          {
            if ( v30 == v37 )
              break;
            if ( v20 != *v37 )
              return v9;
            goto LABEL_29;
          }
          v29 = 1;
        }
        ++v37;
      }
      while ( v30 != v37 );
    }
  }
LABEL_34:
  if ( !(_BYTE)v29 )
  {
    if ( v28 )
      goto LABEL_36;
LABEL_53:
    if ( v32 > 0x1Cu )
      goto LABEL_54;
    goto LABEL_55;
  }
  v67 = v20;
  if ( !sub_98ED70(v20, *(_QWORD *)(a1 + 48), 0, *(_QWORD *)(a1 + 8), 0) )
    return v9;
  v20 = v67;
  v32 = *v67;
LABEL_36:
  if ( v69 && !v70 && (unsigned int)v32 - 12 > 1 )
  {
    v68 = v20;
    if ( !(unsigned __int8)sub_28D2550(a1, a4) )
      return v9;
    v20 = v68;
    v32 = *v68;
  }
  if ( v32 <= 0x1Cu )
    goto LABEL_55;
  v39 = *(_DWORD *)(a1 + 1456);
  v40 = *(_QWORD *)(a1 + 1440);
  if ( v39 )
  {
    v41 = v39 - 1;
    v42 = v41 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v43 = (unsigned __int8 **)(v40 + 16LL * v42);
    v44 = *v43;
    if ( v20 != *v43 )
    {
      for ( j = 1; ; j = v59 )
      {
        if ( v44 == (unsigned __int8 *)-4096LL )
          return v9;
        v59 = j + 1;
        v42 = v41 & (j + v42);
        v43 = (unsigned __int8 **)(v40 + 16LL * v42);
        v44 = *v43;
        if ( v20 == *v43 )
          break;
      }
    }
    v45 = v43[1];
    if ( v45 )
    {
      v46 = (_BYTE *)*((_QWORD *)v45 + 1);
      if ( *v46 > 0x16u )
      {
        v62 = v20;
        v47 = sub_B19DB0(*(_QWORD *)(a1 + 8), (__int64)v46, a4);
        v20 = v62;
        if ( !v47 )
        {
          v48 = *((_QWORD *)v45 + 3);
          if ( !v48 || (v49 = sub_B19DB0(*(_QWORD *)(a1 + 8), v48, a4), v20 = v62, !v49) )
          {
            if ( v45[92] )
              v50 = *((unsigned int *)v45 + 21);
            else
              v50 = *((unsigned int *)v45 + 20);
            v51 = *((_QWORD *)v45 + 9) + 8 * v50;
            v63 = v20;
            v74 = (__int64 *)*((_QWORD *)v45 + 9);
            v75 = v51;
            sub_254BBF0((__int64)&v74);
            v52 = v74;
            v53 = (__int64 *)v75;
            v20 = v63;
            v76 = (__int64 *)(v45 + 64);
            v77 = *((_QWORD *)v45 + 8);
            if ( v74 == (__int64 *)v51 )
              return v9;
            while ( 1 )
            {
              if ( *v52 != *((_QWORD *)v45 + 1) )
              {
                v64 = v20;
                v61 = v53;
                v54 = sub_B19DB0(*(_QWORD *)(a1 + 8), *v52, a4);
                v20 = v64;
                if ( v54 )
                {
                  if ( v52 != (__int64 *)v51 )
                    break;
                  return v9;
                }
                v53 = v61;
              }
              do
                ++v52;
              while ( v52 != v53 && (unsigned __int64)*v52 >= 0xFFFFFFFFFFFFFFFELL );
              if ( v52 == (__int64 *)v51 )
                return v9;
            }
          }
        }
        v32 = *v20;
        goto LABEL_53;
      }
LABEL_54:
      v65 = v20;
      sub_28C8CE0(a1, (__int64)v20);
      v55 = sub_28C8CE0(a1, a4);
      v20 = v65;
      if ( v56 > v55 )
        return v9;
LABEL_55:
      v66 = v20;
      sub_28C79C0(*(_QWORD **)(v9 + 24), *(_DWORD *)(v9 + 32), a1 + 168, v19, v29, (__int64)v20);
      return sub_28CED20(a1, v66);
    }
  }
  return v9;
}
