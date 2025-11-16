// Function: sub_329BA40
// Address: 0x329ba40
//
__int64 __fastcall sub_329BA40(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned __int16 *v10; // rax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r13
  int v18; // eax
  __int64 v19; // r9
  bool v20; // cc
  int v21; // eax
  unsigned __int64 v22; // rdi
  unsigned int v23; // ebx
  unsigned __int64 v24; // r12
  __int64 v25; // rax
  int v27; // eax
  unsigned __int64 v28; // rdi
  int v29; // ebx
  __int16 v30; // ax
  unsigned __int16 *v31; // r14
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rsi
  _QWORD *v37; // rdx
  unsigned int v38; // edx
  __int64 v39; // rdx
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int16 v41; // [rsp+10h] [rbp-80h] BYREF
  __int64 v42; // [rsp+18h] [rbp-78h]
  unsigned __int64 v43; // [rsp+20h] [rbp-70h] BYREF
  __int64 v44; // [rsp+28h] [rbp-68h]
  unsigned __int64 v45; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v46; // [rsp+38h] [rbp-58h]
  unsigned __int64 v47; // [rsp+40h] [rbp-50h]
  unsigned int v48; // [rsp+48h] [rbp-48h]
  int v49; // [rsp+50h] [rbp-40h]
  char v50; // [rsp+54h] [rbp-3Ch]
  __int64 v51; // [rsp+58h] [rbp-38h]

  v5 = a1;
  if ( *(_DWORD *)(a2 + 24) == 216 )
  {
    v25 = *(_QWORD *)(a2 + 40);
    *a4 = *(_QWORD *)v25;
    *((_DWORD *)a4 + 2) = *(_DWORD *)(v25 + 8);
    sub_33DD090(&v45, a1, *a4, a4[1], 0);
    if ( *(_DWORD *)(a5 + 8) > 0x40u && *(_QWORD *)a5 )
      j_j___libc_free_0_0(*(_QWORD *)a5);
    v20 = *(_DWORD *)(a5 + 24) <= 0x40u;
    *(_QWORD *)a5 = v45;
    v27 = (int)v46;
    LODWORD(v46) = 0;
    *(_DWORD *)(a5 + 8) = v27;
    if ( v20 || (v28 = *(_QWORD *)(a5 + 16)) == 0 )
    {
      *(_QWORD *)(a5 + 16) = v47;
      *(_DWORD *)(a5 + 24) = v48;
    }
    else
    {
      j_j___libc_free_0_0(v28);
      v20 = (unsigned int)v46 <= 0x40;
      *(_QWORD *)(a5 + 16) = v47;
      *(_DWORD *)(a5 + 24) = v48;
      if ( !v20 && v45 )
        j_j___libc_free_0_0(v45);
    }
    v29 = *(_BYTE *)(a2 + 28) & 1;
    if ( (*(_BYTE *)(a2 + 28) & 1) == 0 )
      goto LABEL_44;
    v31 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
    v32 = *v31;
    v16 = *((_QWORD *)v31 + 1);
    v41 = v32;
    v42 = v16;
    if ( (_WORD)v32 )
    {
      if ( (unsigned __int16)(v32 - 17) > 0xD3u )
      {
        LOWORD(v43) = v32;
        v44 = v16;
        goto LABEL_55;
      }
      LOWORD(v32) = word_4456580[v32 - 1];
      v39 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v41) )
      {
        v44 = v16;
        LOWORD(v43) = 0;
LABEL_49:
        v45 = sub_3007260((__int64)&v43);
        LODWORD(v36) = v45;
        v46 = v37;
        goto LABEL_50;
      }
      LOWORD(v32) = sub_3009970((__int64)&v41, a1, v33, v34, v35);
    }
    LOWORD(v43) = v32;
    v44 = v39;
    if ( !(_WORD)v32 )
      goto LABEL_49;
LABEL_55:
    if ( (_WORD)v32 == 1 || (unsigned __int16)(v32 - 504) <= 7u )
      BUG();
    v36 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v32 - 16];
LABEL_50:
    v38 = *(_DWORD *)(a5 + 8);
    if ( (_DWORD)v36 != v38 )
    {
      LOBYTE(v16) = v38 <= 0x40 && (unsigned int)v36 <= 0x3F;
      if ( (_BYTE)v16 )
      {
        *(_QWORD *)a5 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v36 - (unsigned __int8)v38 + 64) << v36;
      }
      else
      {
        LODWORD(v16) = v29;
        sub_C43C90((_QWORD *)a5, v36, v38);
      }
      return (unsigned int)v16;
    }
LABEL_44:
    LODWORD(v16) = 1;
    return (unsigned int)v16;
  }
  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v43) = v11;
  v44 = v12;
  if ( (_WORD)v11 )
  {
    if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
      LOWORD(v11) = word_4456580[v11 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v43) )
    {
LABEL_4:
      LODWORD(v16) = 0;
      return (unsigned int)v16;
    }
    v30 = sub_3009970((__int64)&v43, a2, v13, v14, v15);
    v5 = a1;
    LOWORD(v11) = v30;
  }
  if ( (_WORD)v11 != 2 )
    goto LABEL_4;
  v40 = v5;
  LODWORD(v45) = 208;
  v46 = a4;
  v48 = 64;
  v47 = 0;
  v49 = 22;
  v50 = 1;
  v51 = 0;
  v18 = sub_329B960(a2, a3, 0, (__int64)&v45);
  v19 = v40;
  LODWORD(v16) = v18;
  if ( v48 > 0x40 && v47 )
  {
    j_j___libc_free_0_0(v47);
    v19 = v40;
  }
  if ( (_BYTE)v16 )
  {
    sub_33DD090(&v45, v19, *a4, a4[1], 0);
    if ( *(_DWORD *)(a5 + 8) > 0x40u && *(_QWORD *)a5 )
      j_j___libc_free_0_0(*(_QWORD *)a5);
    v20 = *(_DWORD *)(a5 + 24) <= 0x40u;
    *(_QWORD *)a5 = v45;
    v21 = (int)v46;
    LODWORD(v46) = 0;
    *(_DWORD *)(a5 + 8) = v21;
    if ( v20 || (v22 = *(_QWORD *)(a5 + 16)) == 0 )
    {
      *(_QWORD *)(a5 + 16) = v47;
      *(_DWORD *)(a5 + 24) = v48;
    }
    else
    {
      j_j___libc_free_0_0(v22);
      v20 = (unsigned int)v46 <= 0x40;
      *(_QWORD *)(a5 + 16) = v47;
      *(_DWORD *)(a5 + 24) = v48;
      if ( !v20 && v45 )
      {
        j_j___libc_free_0_0(v45);
        v23 = *(_DWORD *)(a5 + 8);
        LODWORD(v44) = v23;
        if ( v23 > 0x40 )
          goto LABEL_21;
        goto LABEL_30;
      }
    }
    v23 = *(_DWORD *)(a5 + 8);
    LODWORD(v44) = v23;
    if ( v23 > 0x40 )
    {
LABEL_21:
      sub_C43780((__int64)&v43, (const void **)a5);
      v23 = v44;
      if ( (unsigned int)v44 > 0x40 )
      {
        *(_QWORD *)v43 |= 1uLL;
        v23 = v44;
        v24 = v43;
        LODWORD(v44) = 0;
        LODWORD(v46) = v23;
        v45 = v43;
        if ( !v23 )
          return (unsigned int)v16;
        if ( v23 > 0x40 )
        {
          LOBYTE(v16) = v23 == (unsigned int)sub_C445E0((__int64)&v45);
          if ( v24 )
          {
            j_j___libc_free_0_0(v24);
            if ( (unsigned int)v44 > 0x40 )
            {
              if ( v43 )
                j_j___libc_free_0_0(v43);
            }
          }
          return (unsigned int)v16;
        }
LABEL_33:
        LOBYTE(v16) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) == v24;
        return (unsigned int)v16;
      }
LABEL_31:
      if ( !v23 )
        return (unsigned int)v16;
      v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & (v43 | 1);
      goto LABEL_33;
    }
LABEL_30:
    v43 = *(_QWORD *)a5;
    goto LABEL_31;
  }
  return (unsigned int)v16;
}
