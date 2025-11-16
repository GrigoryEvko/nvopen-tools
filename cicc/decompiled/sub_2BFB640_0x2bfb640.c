// Function: sub_2BFB640
// Address: 0x2bfb640
//
__int64 __fastcall sub_2BFB640(__int64 a1, __int64 a2, char a3)
{
  unsigned int v4; // r9d
  __int64 v5; // rcx
  unsigned int v6; // r10d
  int v7; // r12d
  unsigned int v8; // edx
  unsigned int v9; // ebx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r11
  __int64 v13; // r15
  bool v15; // r13
  char v16; // al
  unsigned int v17; // edx
  bool v18; // bl
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdi
  __int64 v32; // rsi
  char v33; // al
  __int64 v34; // rax
  unsigned int v35; // ebx
  int v36; // ecx
  char v37; // dl
  int v38; // ebx
  __int64 *v39; // r11
  int v40; // eax
  int v41; // edx
  __int64 v42; // rax
  int v43; // esi
  __int16 v44; // [rsp+Eh] [rbp-92h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+38h] [rbp-68h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-58h] BYREF
  char v50; // [rsp+4Ch] [rbp-54h]
  __int64 v51; // [rsp+50h] [rbp-50h]
  __int64 v52; // [rsp+58h] [rbp-48h] BYREF
  __int64 v53[8]; // [rsp+60h] [rbp-40h] BYREF

  v48 = a2;
  if ( a3 )
  {
    LODWORD(v53[0]) = 0;
    BYTE4(v53[0]) = 0;
    return sub_2BFB120(a1, a2, (unsigned int *)v53);
  }
  v4 = *(_DWORD *)(a1 + 56);
  v5 = *(_QWORD *)(a1 + 40);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = 1;
    v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = v8;
    v10 = (__int64 *)(v5 + 16LL * v8);
    v11 = *v10;
    v12 = *v10;
    if ( *v10 == a2 )
      return v10[1];
    while ( 1 )
    {
      if ( v12 == -4096 )
        goto LABEL_9;
      v9 = v6 & (v7 + v9);
      v12 = *(_QWORD *)(v5 + 16LL * v9);
      if ( v12 == a2 )
        break;
      ++v7;
    }
    v38 = 1;
    v39 = 0;
    while ( v11 != -4096 )
    {
      if ( !v39 && v11 == -8192 )
        v39 = v10;
      v8 = v6 & (v38 + v8);
      v10 = (__int64 *)(v5 + 16LL * v8);
      v11 = *v10;
      if ( *v10 == a2 )
        return v10[1];
      ++v38;
    }
    if ( !v39 )
      v39 = v10;
    v40 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    v41 = v40 + 1;
    v53[0] = (__int64)v39;
    if ( 4 * (v40 + 1) >= 3 * v4 )
    {
      v43 = 2 * v4;
    }
    else
    {
      if ( v4 - *(_DWORD *)(a1 + 52) - v41 > v4 >> 3 )
      {
LABEL_49:
        *(_DWORD *)(a1 + 48) = v41;
        if ( *v39 != -4096 )
          --*(_DWORD *)(a1 + 52);
        *v39 = a2;
        v13 = 0;
        v39[1] = 0;
        return v13;
      }
      v43 = v4;
    }
    sub_2AC6AB0(a1 + 32, v43);
    sub_2ABE290(a1 + 32, &v48, v53);
    a2 = v48;
    v39 = (__int64 *)v53[0];
    v41 = *(_DWORD *)(a1 + 48) + 1;
    goto LABEL_49;
  }
LABEL_9:
  v53[0] = a1;
  v53[1] = a2;
  v15 = sub_2BEF470(a1, a2, 0, 0);
  if ( !v15 )
  {
    v13 = sub_2BF4520(v53, *(_QWORD *)(a2 + 40));
LABEL_31:
    sub_2BF26E0(a1, v48, v13, 0);
    return v13;
  }
  LODWORD(v52) = 0;
  BYTE4(v52) = 0;
  v13 = sub_2BFB120(a1, a2, (unsigned int *)&v52);
  if ( !*(_BYTE *)(a1 + 12) && *(_DWORD *)(a1 + 8) == 1 )
    goto LABEL_31;
  v16 = sub_2AAA120(v48);
  v17 = 0;
  v18 = v16;
  if ( !v16 )
    v17 = *(_DWORD *)(a1 + 8) - 1;
  v49 = v17;
  v50 = 0;
  if ( !sub_2BEF470(a1, v48, v17, 0) )
  {
    v49 = 0;
    v18 = v15;
  }
  v19 = sub_2BFB120(a1, v48, &v49);
  v20 = *(_QWORD *)(a1 + 904);
  v21 = v19;
  v22 = *(_QWORD *)(v20 + 56);
  v44 = *(_WORD *)(v20 + 64);
  v47 = *(_QWORD *)(v20 + 48);
  if ( *(_BYTE *)v19 == 84 )
  {
    v46 = v19;
    v42 = sub_AA4FF0(*(_QWORD *)(v19 + 40));
    v20 = *(_QWORD *)(a1 + 904);
    v21 = v46;
    v23 = v42;
  }
  else
  {
    v23 = *(_QWORD *)(v19 + 32);
  }
  v45 = v21;
  if ( v23 )
    v23 -= 24;
  sub_D5F1F0(v20, v23);
  if ( v18 )
  {
    v13 = sub_2BF4520(v53, v13);
    sub_2BF26E0(a1, v48, v13, 0);
  }
  else
  {
    v31 = *(_QWORD *)(v45 + 8);
    v32 = *(_QWORD *)(a1 + 8);
    v33 = *(_BYTE *)(v31 + 8);
    v51 = v32;
    if ( v33 == 15 )
    {
      v31 = (__int64)sub_E454C0(v31, v32, v24, v45, v25, v26);
    }
    else
    {
      v36 = *(_DWORD *)(a1 + 8);
      v37 = *(_BYTE *)(a1 + 12);
      v52 = v32;
      if ( ((v33 - 7) & 0xFD) != 0 && (v37 || v36 != 1) )
        v31 = sub_BCE1B0((__int64 *)v31, v52);
    }
    v34 = sub_ACADE0((__int64 **)v31);
    v35 = 0;
    sub_2BF26E0(a1, v48, v34, 0);
    if ( *(_DWORD *)(a1 + 8) )
    {
      do
      {
        LODWORD(v52) = v35;
        BYTE4(v52) = 0;
        ++v35;
        sub_2BFBAC0(a1, v48, &v52);
      }
      while ( v35 < *(_DWORD *)(a1 + 8) );
    }
    v13 = sub_2BFB640(a1, v48, 0);
  }
  v27 = *(_QWORD *)(a1 + 904);
  if ( v47 )
  {
    *(_QWORD *)(v27 + 48) = v47;
    *(_QWORD *)(v27 + 56) = v22;
    *(_WORD *)(v27 + 64) = v44;
    if ( v22 != v47 + 48 )
    {
      if ( v22 )
        v22 -= 24;
      v29 = *(_QWORD *)sub_B46C60(v22);
      v52 = v29;
      if ( v29 )
      {
        sub_B96E90((__int64)&v52, v29, 1);
        v29 = v52;
      }
      sub_F80810(v27, 0, v29, v28, v29, v30);
      if ( v52 )
        sub_B91220((__int64)&v52, v52);
    }
  }
  else
  {
    *(_QWORD *)(v27 + 48) = 0;
    *(_QWORD *)(v27 + 56) = 0;
    *(_WORD *)(v27 + 64) = 0;
  }
  return v13;
}
