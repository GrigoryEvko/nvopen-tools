// Function: sub_1124A20
// Address: 0x1124a20
//
__int64 __fastcall sub_1124A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  const char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  unsigned int v17; // esi
  unsigned int v18; // eax
  __int16 v19; // ax
  __int64 v20; // r13
  __int64 v21; // r13
  _QWORD *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 i; // r13
  char *v27; // rax
  unsigned __int8 v28; // dl
  __int64 v30; // rdi
  unsigned int v31; // edx
  unsigned __int64 v32; // rax
  unsigned int v33; // edx
  unsigned __int64 v34; // rax
  const char *v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r12
  _BYTE *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // r10
  __int64 v44; // rax
  unsigned __int8 *v45; // r15
  int v46; // eax
  _QWORD *v47; // rax
  _QWORD *v48; // r10
  __int64 v49; // r11
  unsigned int *v50; // rcx
  __int64 v51; // r14
  __int64 v52; // r13
  unsigned int *v53; // rbx
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-F8h]
  __int64 v60; // [rsp+10h] [rbp-F0h]
  unsigned int v61; // [rsp+18h] [rbp-E8h]
  __int64 v62; // [rsp+18h] [rbp-E8h]
  _QWORD *v63; // [rsp+18h] [rbp-E8h]
  __int64 v64; // [rsp+18h] [rbp-E8h]
  __int64 v65; // [rsp+18h] [rbp-E8h]
  __int64 v66; // [rsp+20h] [rbp-E0h]
  _QWORD *v67; // [rsp+28h] [rbp-D8h]
  __int64 v68; // [rsp+28h] [rbp-D8h]
  unsigned int v69; // [rsp+38h] [rbp-C8h]
  __int64 v70; // [rsp+40h] [rbp-C0h]
  unsigned int v71; // [rsp+40h] [rbp-C0h]
  __int64 v73; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v74; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v75; // [rsp+68h] [rbp-98h]
  unsigned __int64 v76; // [rsp+70h] [rbp-90h] BYREF
  __int64 v77; // [rsp+78h] [rbp-88h]
  __int16 v78; // [rsp+90h] [rbp-70h]
  unsigned __int64 v79; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-58h]
  __int16 v81; // [rsp+C0h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 12 || *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *v7;
  v9 = *(_QWORD *)(v7[4] - 32);
  v67 = *(_QWORD **)(v8 - 32);
  v10 = *(_QWORD *)(v9 + 8);
  v66 = v9;
  v70 = v67[1];
  v11 = (const char *)sub_BCAE30(v70);
  v80 = v12;
  v79 = (unsigned __int64)v11;
  v69 = sub_CA1930(&v79);
  v79 = sub_BCAE30(v10);
  v80 = v13;
  v14 = sub_CA1930(&v79);
  v15 = v70;
  v17 = v14;
  v61 = v14;
  v18 = v69;
  if ( v69 < v17 )
  {
    v18 = v17;
    v15 = v10;
  }
  v71 = v18;
  v73 = v15;
  if ( (unsigned __int8)sub_BD3660(a2, 2) )
  {
    for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
    {
      v27 = *(char **)(i + 24);
      if ( v27 != (char *)a1 )
      {
        v28 = *v27;
        if ( (unsigned __int8)*v27 <= 0x1Cu )
          return 0;
        if ( v28 == 67 )
        {
          v79 = sub_BCAE30(*((_QWORD *)v27 + 1));
          v80 = v36;
          if ( v71 < (unsigned int)sub_CA1930(&v79) )
            return 0;
        }
        else
        {
          if ( v28 != 57 )
            return 0;
          v30 = *((_QWORD *)v27 - 4);
          if ( *(_BYTE *)v30 != 17 )
            return 0;
          v31 = *(_DWORD *)(v30 + 32);
          if ( v31 > 0x40 )
          {
            v33 = v31 - sub_C444A0(v30 + 24);
LABEL_39:
            if ( v71 < v33 )
              return 0;
            continue;
          }
          v32 = *(_QWORD *)(v30 + 24);
          if ( v32 )
          {
            _BitScanReverse64(&v32, v32);
            v33 = 64 - (v32 ^ 0x3F);
            goto LABEL_39;
          }
        }
      }
    }
  }
  v19 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( v19 == 34 )
  {
    LODWORD(v77) = v71;
    if ( v71 > 0x40 )
    {
      sub_C43690((__int64)&v76, -1, 1);
    }
    else
    {
      v34 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v71;
      if ( !v71 )
        v34 = 0;
      v76 = v34;
    }
    sub_C449B0((__int64)&v79, (const void **)&v76, *(_DWORD *)(a3 + 8));
    if ( (unsigned int)v77 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
    v35 = (const char *)v79;
    v76 = v79;
    LODWORD(v77) = v80;
    if ( (unsigned int)v80 <= 0x40 )
    {
      if ( *(_QWORD *)a3 == v79 )
        goto LABEL_14;
    }
    else if ( sub_C43C50((__int64)&v76, (const void **)a3) )
    {
      if ( v35 )
        j_j___libc_free_0_0(v35);
      goto LABEL_14;
    }
    sub_969240((__int64 *)&v76);
    return 0;
  }
  if ( v19 != 36 )
    return 0;
  LODWORD(v80) = *(_DWORD *)(a3 + 8);
  v20 = 1LL << v71;
  if ( (unsigned int)v80 <= 0x40 )
  {
    v79 = 0;
LABEL_12:
    v79 |= v20;
    goto LABEL_13;
  }
  sub_C43690((__int64)&v79, 0, 0);
  if ( (unsigned int)v80 <= 0x40 )
    goto LABEL_12;
  *(_QWORD *)(v79 + 8LL * (v71 >> 6)) |= v20;
  if ( (unsigned int)v80 > 0x40 )
  {
    if ( sub_C43C50((__int64)&v79, (const void **)a3) )
    {
      if ( v79 )
        j_j___libc_free_0_0(v79);
      goto LABEL_14;
    }
LABEL_94:
    v22 = 0;
    sub_969240((__int64 *)&v79);
    return (__int64)v22;
  }
LABEL_13:
  if ( v79 != *(_QWORD *)a3 )
    goto LABEL_94;
LABEL_14:
  v21 = *(_QWORD *)(a4 + 32);
  sub_D5F1F0(v21, a2);
  if ( v71 > v69 )
  {
    v81 = 257;
    v67 = (_QWORD *)sub_A82F30((unsigned int **)v21, (__int64)v67, v73, (__int64)&v79, 0);
  }
  if ( v71 > v61 )
  {
    v81 = 257;
    v66 = sub_A82F30((unsigned int **)v21, v66, v73, (__int64)&v79, 0);
  }
  v79 = (unsigned __int64)"umul";
  BYTE4(v74) = 0;
  v76 = (unsigned __int64)v67;
  v81 = 259;
  v77 = v66;
  v68 = sub_B33D10(v21, 0x171u, (__int64)&v73, 1, (int)&v76, 2, v74, (__int64)&v79);
  sub_F15FC0(*(_QWORD *)(a4 + 40), a2);
  if ( (unsigned __int8)sub_BD3660(a2, 2) )
  {
    v81 = 259;
    v79 = (unsigned __int64)"umul.value";
    LODWORD(v76) = 0;
    v38 = sub_94D3D0((unsigned int **)v21, v68, (__int64)&v76, 1, (__int64)&v79);
    v39 = *(_QWORD *)(a2 + 16);
    v60 = v38;
    while ( v39 )
    {
      v44 = v39;
      v39 = *(_QWORD *)(v39 + 8);
      v45 = *(unsigned __int8 **)(v44 + 24);
      if ( v45 != (unsigned __int8 *)a1 )
      {
        v46 = *v45;
        if ( (unsigned __int8)v46 <= 0x1Cu )
LABEL_97:
          BUG();
        if ( (_BYTE)v46 == 67 )
        {
          v79 = sub_BCAE30(*((_QWORD *)v45 + 1));
          v80 = v56;
          if ( sub_CA1930(&v79) == v71 )
          {
            sub_F162A0(a4, (__int64)v45, v60);
          }
          else
          {
            if ( *((_QWORD *)v45 - 4) )
            {
              v57 = *((_QWORD *)v45 - 3);
              **((_QWORD **)v45 - 2) = v57;
              if ( v57 )
                *(_QWORD *)(v57 + 16) = *((_QWORD *)v45 - 2);
            }
            *((_QWORD *)v45 - 4) = v60;
            if ( v60 )
            {
              v58 = *(_QWORD *)(v60 + 16);
              *((_QWORD *)v45 - 3) = v58;
              if ( v58 )
                *(_QWORD *)(v58 + 16) = v45 - 24;
              *((_QWORD *)v45 - 2) = v60 + 16;
              *(_QWORD *)(v60 + 16) = v45 - 32;
            }
          }
        }
        else
        {
          if ( (unsigned int)(v46 - 42) > 0x11 )
            goto LABEL_97;
          sub_C44740((__int64)&v74, (char **)(*((_QWORD *)v45 - 4) + 24LL), v71);
          v81 = 257;
          v40 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v60 + 8), (__int64)&v74);
          v41 = sub_A82350((unsigned int **)v21, (_BYTE *)v60, v40, (__int64)&v79);
          v78 = 257;
          v42 = v41;
          if ( *((_QWORD *)v45 + 1) == *(_QWORD *)(v41 + 8) )
          {
            v43 = v41;
          }
          else
          {
            v62 = *((_QWORD *)v45 + 1);
            v43 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v21 + 80) + 120LL))(
                    *(_QWORD *)(v21 + 80),
                    39,
                    v41,
                    v62);
            if ( !v43 )
            {
              v81 = 257;
              v47 = sub_BD2C40(72, unk_3F10A14);
              v48 = v47;
              if ( v47 )
              {
                v49 = v62;
                v63 = v47;
                sub_B515B0((__int64)v47, v42, v49, (__int64)&v79, 0, 0);
                v48 = v63;
              }
              v64 = (__int64)v48;
              (*(void (__fastcall **)(_QWORD, _QWORD *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v21 + 88)
                                                                                           + 16LL))(
                *(_QWORD *)(v21 + 88),
                v48,
                &v76,
                *(_QWORD *)(v21 + 56),
                *(_QWORD *)(v21 + 64));
              v50 = *(unsigned int **)v21;
              v43 = v64;
              if ( *(_QWORD *)v21 != *(_QWORD *)v21 + 16LL * *(unsigned int *)(v21 + 8) )
              {
                v65 = v21;
                v51 = *(_QWORD *)v21 + 16LL * *(unsigned int *)(v21 + 8);
                v52 = v43;
                v59 = a4;
                v53 = v50;
                do
                {
                  v54 = *((_QWORD *)v53 + 1);
                  v55 = *v53;
                  v53 += 4;
                  sub_B99FD0(v52, v55, v54);
                }
                while ( (unsigned int *)v51 != v53 );
                v43 = v52;
                a4 = v59;
                v21 = v65;
              }
            }
          }
          sub_F162A0(a4, (__int64)v45, v43);
          if ( v75 > 0x40 && v74 )
            j_j___libc_free_0_0(v74);
        }
        sub_F15FC0(*(_QWORD *)(a4 + 40), (__int64)v45);
      }
    }
  }
  LODWORD(v76) = 1;
  v81 = 257;
  if ( (*(_WORD *)(a1 + 2) & 0x3F) == 0x24 )
  {
    v37 = sub_94D3D0((unsigned int **)v21, v68, (__int64)&v76, 1, (__int64)&v79);
    v81 = 257;
    return sub_B50640(v37, (__int64)&v79, 0, 0);
  }
  else
  {
    v22 = sub_BD2C40(104, unk_3F10A14);
    if ( v22 )
    {
      v23 = sub_B501B0(*(_QWORD *)(v68 + 8), (unsigned int *)&v76, 1);
      sub_B44260((__int64)v22, v23, 64, 1u, 0, 0);
      if ( *(v22 - 4) )
      {
        v24 = *(v22 - 3);
        *(_QWORD *)*(v22 - 2) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = *(v22 - 2);
      }
      *(v22 - 4) = v68;
      v25 = *(_QWORD *)(v68 + 16);
      *(v22 - 3) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = v22 - 3;
      *(v22 - 2) = v68 + 16;
      *(_QWORD *)(v68 + 16) = v22 - 4;
      v22[9] = v22 + 11;
      v22[10] = 0x400000000LL;
      sub_B50030((__int64)v22, &v76, 1, (__int64)&v79);
    }
  }
  return (__int64)v22;
}
