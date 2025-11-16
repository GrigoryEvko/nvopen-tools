// Function: sub_D0CFC0
// Address: 0xd0cfc0
//
__int64 __fastcall sub_D0CFC0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // r13
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // r12d
  __int64 *v23; // rax
  __int64 *v24; // r15
  __int64 v25; // rsi
  _QWORD *v26; // rax
  char v27; // dl
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // r15
  _QWORD *v32; // rax
  _BYTE *v33; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // r15
  int v38; // r14d
  __int64 v39; // rax
  int v40; // ecx
  unsigned int v41; // r14d
  __int64 *v42; // rbx
  int v43; // r13d
  __int64 *v44; // r14
  __int64 *v45; // r15
  _QWORD *v46; // rdx
  __int64 v47; // rcx
  _QWORD *v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r9
  _QWORD *v51; // rax
  __int64 *v52; // rax
  const void *v53; // [rsp+0h] [rbp-220h]
  __int64 v54; // [rsp+8h] [rbp-218h]
  __int64 v55; // [rsp+18h] [rbp-208h]
  int v56; // [rsp+20h] [rbp-200h]
  __int64 v57; // [rsp+28h] [rbp-1F8h]
  __int64 v60; // [rsp+40h] [rbp-1E0h] BYREF
  _QWORD *v61; // [rsp+48h] [rbp-1D8h]
  __int64 v62; // [rsp+50h] [rbp-1D0h]
  int v63; // [rsp+58h] [rbp-1C8h]
  char v64; // [rsp+5Ch] [rbp-1C4h]
  _QWORD v65[2]; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v66; // [rsp+70h] [rbp-1B0h] BYREF
  _BYTE *v67; // [rsp+78h] [rbp-1A8h]
  __int64 v68; // [rsp+80h] [rbp-1A0h]
  int v69; // [rsp+88h] [rbp-198h]
  char v70; // [rsp+8Ch] [rbp-194h]
  _BYTE v71[64]; // [rsp+90h] [rbp-190h] BYREF
  __int64 v72; // [rsp+D0h] [rbp-150h] BYREF
  __int64 *v73; // [rsp+D8h] [rbp-148h]
  __int64 v74; // [rsp+E0h] [rbp-140h]
  int v75; // [rsp+E8h] [rbp-138h]
  unsigned __int8 v76; // [rsp+ECh] [rbp-134h]
  char v77; // [rsp+F0h] [rbp-130h] BYREF

  v57 = a4;
  if ( !a4 )
    goto LABEL_7;
  v8 = *a2;
  if ( *a2 )
  {
    v9 = a4;
    v10 = (unsigned int)(*(_DWORD *)(v8 + 44) + 1);
    if ( (unsigned int)(*(_DWORD *)(v8 + 44) + 1) < *(_DWORD *)(a4 + 32) )
    {
LABEL_4:
      v11 = 0;
      if ( *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v10) )
        v11 = v9;
      v57 = v11;
LABEL_7:
      if ( a3 )
        goto LABEL_8;
LABEL_34:
      v66 = 0;
      v67 = v71;
      v68 = 8;
      v69 = 0;
      v70 = 1;
      v60 = 0;
      v61 = v65;
      v62 = 2;
      v63 = 0;
      v64 = 1;
      if ( !a5 )
        goto LABEL_12;
      goto LABEL_30;
    }
  }
  else
  {
    v9 = a4;
    v10 = 0;
    if ( *(_DWORD *)(a4 + 32) )
      goto LABEL_4;
  }
  v57 = 0;
  if ( !a3 )
    goto LABEL_34;
LABEL_8:
  v12 = *(unsigned int *)(a3 + 20);
  v13 = 0;
  v14 = *(_DWORD *)(a3 + 24) == (_DWORD)v12;
  v66 = 0;
  if ( v14 )
    v13 = v57;
  v68 = 8;
  v69 = 0;
  v57 = v13;
  v67 = v71;
  v70 = 1;
  if ( a5 )
  {
    v23 = *(__int64 **)(a3 + 8);
    if ( !*(_BYTE *)(a3 + 28) )
      v12 = *(unsigned int *)(a3 + 16);
    v24 = &v23[v12];
    if ( v23 != v24 )
    {
      while ( 1 )
      {
        v25 = *v23;
        if ( (unsigned __int64)*v23 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v24 == ++v23 )
          goto LABEL_29;
      }
      if ( v24 != v23 )
      {
        v44 = v24;
        v45 = v23;
        while ( 1 )
        {
          v48 = sub_D0CF30(a5, v25);
          if ( !v48 )
            goto LABEL_100;
          if ( !v70 )
            break;
          v51 = v67;
          v46 = &v67[8 * HIDWORD(v68)];
          if ( v67 == (_BYTE *)v46 )
          {
LABEL_107:
            if ( HIDWORD(v68) >= (unsigned int)v68 )
              break;
            ++HIDWORD(v68);
            *v46 = v48;
            ++v66;
          }
          else
          {
            while ( v48 != (_QWORD *)*v51 )
            {
              if ( v46 == ++v51 )
                goto LABEL_107;
            }
          }
LABEL_100:
          v52 = v45 + 1;
          if ( v45 + 1 == v44 )
            goto LABEL_29;
          v25 = *v52;
          for ( ++v45; (unsigned __int64)*v52 >= 0xFFFFFFFFFFFFFFFELL; v45 = v52 )
          {
            if ( v44 == ++v52 )
              goto LABEL_29;
            v25 = *v52;
          }
          if ( v44 == v45 )
            goto LABEL_29;
        }
        sub_C8CC70((__int64)&v66, (__int64)v48, (__int64)v46, v47, v49, v50);
        goto LABEL_100;
      }
    }
LABEL_29:
    v60 = 0;
    v61 = v65;
    v62 = 2;
    v63 = 0;
    v64 = 1;
LABEL_30:
    v26 = sub_D0CF30(a5, *a2);
    if ( v26 )
    {
      HIDWORD(v62) = 1;
      v65[0] = v26;
      v60 = 1;
    }
    goto LABEL_12;
  }
  v60 = 0;
  v61 = v65;
  v62 = 2;
  v63 = 0;
  v64 = 1;
LABEL_12:
  v72 = 0;
  v15 = 1;
  v74 = 32;
  v56 = qword_4F86A48;
  v73 = (__int64 *)&v77;
  v75 = 0;
  v76 = 1;
  v53 = (const void *)(a1 + 16);
  v16 = *(_DWORD *)(a1 + 8);
  while ( 1 )
  {
    v17 = *(__int64 **)a1;
    v18 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * v16 - 8);
    *(_DWORD *)(a1 + 8) = v16 - 1;
    if ( !(_BYTE)v15 )
      goto LABEL_36;
    v19 = v73;
    v20 = HIDWORD(v74);
    v17 = &v73[HIDWORD(v74)];
    if ( v73 != v17 )
    {
      while ( v18 != *v19 )
      {
        if ( v17 == ++v19 )
          goto LABEL_65;
      }
      goto LABEL_18;
    }
LABEL_65:
    if ( HIDWORD(v74) < (unsigned int)v74 )
    {
      v20 = (unsigned int)++HIDWORD(v74);
      *v17 = v18;
      v15 = v76;
      ++v72;
    }
    else
    {
LABEL_36:
      v20 = v18;
      sub_C8CC70((__int64)&v72, v18, (__int64)v17, v15, a5, a6);
      v15 = v76;
      if ( !v27 )
        goto LABEL_18;
    }
    if ( v18 == *a2 )
      goto LABEL_61;
    if ( !a3 )
      break;
    if ( *(_BYTE *)(a3 + 28) )
    {
      v28 = *(_QWORD **)(a3 + 8);
      v29 = &v28[*(unsigned int *)(a3 + 20)];
      if ( v28 == v29 )
        break;
      while ( v18 != *v28 )
      {
        if ( v29 == ++v28 )
          goto LABEL_46;
      }
    }
    else
    {
      v20 = v18;
      if ( !sub_C8CA60(a3, v18) )
        break;
      v15 = v76;
    }
LABEL_18:
    v16 = *(_DWORD *)(a1 + 8);
LABEL_19:
    if ( !v16 )
    {
      v21 = 0;
      if ( !(_BYTE)v15 )
        goto LABEL_62;
LABEL_21:
      if ( !v64 )
        goto LABEL_63;
LABEL_22:
      if ( v70 )
        return v21;
      goto LABEL_64;
    }
  }
LABEL_46:
  if ( !v57 || (v20 = v18, !(unsigned __int8)sub_B19720(v57, v18, *a2)) )
  {
    if ( a5 )
      goto LABEL_49;
LABEL_74:
    if ( !--v56 )
      goto LABEL_60;
LABEL_75:
    v36 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v36 == v18 + 48 )
      goto LABEL_90;
    if ( !v36 )
      BUG();
    v37 = v36 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 > 0xA )
    {
LABEL_90:
      v55 = 0;
      v38 = 0;
      v37 = 0;
    }
    else
    {
      v38 = sub_B46E30(v37);
      v55 = v38;
    }
    v39 = *(unsigned int *)(a1 + 8);
    if ( v39 + v55 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v20 = (__int64)v53;
      sub_C8D5F0(a1, v53, v39 + v55, 8u, a5, a6);
      v39 = *(unsigned int *)(a1 + 8);
    }
    v40 = v38;
    if ( v38 )
    {
      v54 = a1;
      v41 = 0;
      v42 = (__int64 *)(*(_QWORD *)a1 + 8 * v39);
      v43 = v40;
      do
      {
        if ( v42 )
        {
          v20 = v41;
          *v42 = sub_B46EC0(v37, v41);
        }
        ++v41;
        ++v42;
      }
      while ( v41 != v43 );
      a1 = v54;
      LODWORD(v39) = *(_DWORD *)(v54 + 8);
    }
    *(_DWORD *)(a1 + 8) = v39 + v55;
    v16 = v39 + v55;
    goto LABEL_71;
  }
  if ( a2 + 1 != a2 )
    goto LABEL_60;
  if ( !a5 )
    goto LABEL_74;
LABEL_49:
  v20 = v18;
  v30 = sub_D0CF30(a5, v18);
  v31 = (__int64)v30;
  if ( v70 )
  {
    v32 = v67;
    v33 = &v67[8 * HIDWORD(v68)];
    if ( v67 == v33 )
      goto LABEL_55;
    while ( v31 != *v32 )
    {
      if ( v33 == (_BYTE *)++v32 )
        goto LABEL_55;
    }
LABEL_54:
    v31 = 0;
    goto LABEL_55;
  }
  v20 = (__int64)v30;
  if ( sub_C8CA60((__int64)&v66, (__int64)v30) )
    goto LABEL_54;
LABEL_55:
  if ( !v64 )
  {
    v20 = v31;
    if ( sub_C8CA60((__int64)&v60, v31) )
      goto LABEL_60;
    goto LABEL_68;
  }
  v34 = v61;
  v35 = &v61[HIDWORD(v62)];
  if ( v61 == v35 )
  {
LABEL_68:
    if ( !--v56 )
      goto LABEL_60;
    if ( !v31 )
      goto LABEL_75;
    v20 = a1;
    sub_D472F0(v31, a1);
    v16 = *(_DWORD *)(a1 + 8);
LABEL_71:
    v15 = v76;
    goto LABEL_19;
  }
  while ( v31 != *v34 )
  {
    if ( v35 == ++v34 )
      goto LABEL_68;
  }
LABEL_60:
  LOBYTE(v15) = v76;
LABEL_61:
  v21 = 1;
  if ( (_BYTE)v15 )
    goto LABEL_21;
LABEL_62:
  _libc_free(v73, v20);
  if ( v64 )
    goto LABEL_22;
LABEL_63:
  _libc_free(v61, v20);
  if ( v70 )
    return v21;
LABEL_64:
  _libc_free(v67, v20);
  return v21;
}
