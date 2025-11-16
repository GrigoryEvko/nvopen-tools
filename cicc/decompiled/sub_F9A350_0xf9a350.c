// Function: sub_F9A350
// Address: 0xf9a350
//
__int64 __fastcall sub_F9A350(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // eax
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned int v17; // r12d
  __int64 v19; // r10
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 i; // r14
  int v27; // ebx
  __int64 v28; // rax
  int v29; // r13d
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 j; // r14
  unsigned int v35; // ebx
  __int64 v36; // rax
  int v37; // r13d
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rdi
  const char *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // eax
  __int64 *v46; // rax
  unsigned int *v47; // rbx
  __int64 v48; // r13
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // rdx
  char *v52; // rax
  unsigned int v53; // ecx
  char *v54; // r9
  unsigned __int64 v55; // rdx
  __int64 v56; // r8
  __int64 v57; // [rsp+0h] [rbp-220h]
  _BYTE *v58; // [rsp+8h] [rbp-218h]
  __int64 v59; // [rsp+10h] [rbp-210h]
  __int64 v60; // [rsp+10h] [rbp-210h]
  __int64 v61; // [rsp+10h] [rbp-210h]
  char v62; // [rsp+1Fh] [rbp-201h]
  char *v63; // [rsp+20h] [rbp-200h]
  _QWORD **v64; // [rsp+28h] [rbp-1F8h]
  __int64 v65; // [rsp+28h] [rbp-1F8h]
  __int64 v67; // [rsp+48h] [rbp-1D8h]
  __int64 v68; // [rsp+48h] [rbp-1D8h]
  __int64 v69; // [rsp+48h] [rbp-1D8h]
  _QWORD v70[4]; // [rsp+50h] [rbp-1D0h] BYREF
  __int16 v71; // [rsp+70h] [rbp-1B0h]
  char *v72; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v73; // [rsp+88h] [rbp-198h]
  _BYTE v74[16]; // [rsp+90h] [rbp-190h] BYREF
  __int16 v75; // [rsp+A0h] [rbp-180h]
  _QWORD *v76; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-148h]
  _BYTE v78[128]; // [rsp+E0h] [rbp-140h] BYREF
  _QWORD *v79; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+168h] [rbp-B8h]
  _BYTE v81[176]; // [rsp+170h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 - 8);
  v6 = 1;
  v7 = sub_AA5030(*(_QWORD *)(v5 + 32), 1);
  if ( !v7 )
    BUG();
  v62 = *(_BYTE *)(v7 - 24);
  v63 = *(char **)(a2 + 40);
  if ( v62 == 36 )
    v67 = 0;
  else
    v67 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
  v76 = v78;
  v77 = 0x1000000000LL;
  v80 = 0x1000000000LL;
  v8 = *(_DWORD *)(a2 + 4);
  v79 = v81;
  v9 = (v8 & 0x7FFFFFFu) >> 1;
  v10 = v9 - 1;
  if ( v9 == 1 )
    return 0;
  v11 = v67;
  v12 = 1;
  v13 = 0;
  do
  {
    v14 = 32;
    if ( (_DWORD)v12 != -1 )
      v14 = 32LL * (unsigned int)(2 * v12 + 1);
    v15 = *(_QWORD *)(a2 - 8);
    v16 = *(_QWORD *)(v15 + v14);
    if ( !v11 || v16 == v11 )
    {
      v19 = v12;
      v20 = *(_QWORD *)(v15 + 32LL * (unsigned int)(2 * v12));
      v21 = (unsigned int)v77;
      v22 = (unsigned int)v77 + 1LL;
      if ( v22 > HIDWORD(v77) )
      {
        v6 = (unsigned __int64)v78;
        v60 = v10;
        v69 = *(_QWORD *)(v15 + 32LL * (unsigned int)(2 * v12));
        sub_C8D5F0((__int64)&v76, v78, v22, 8u, v10, v20);
        v21 = (unsigned int)v77;
        v10 = v60;
        v19 = v12;
        v20 = v69;
      }
      v76[v21] = v20;
      v11 = v16;
      LODWORD(v77) = v77 + 1;
    }
    else
    {
      if ( v13 && v16 != v13 )
        goto LABEL_12;
      v19 = v12;
      v31 = *(_QWORD *)(v15 + 32LL * (unsigned int)(2 * v12));
      v32 = (unsigned int)v80;
      v33 = (unsigned int)v80 + 1LL;
      if ( v33 > HIDWORD(v80) )
      {
        v6 = (unsigned __int64)v81;
        v61 = v10;
        v65 = v11;
        sub_C8D5F0((__int64)&v79, v81, v33, 8u, v10, v11);
        v32 = (unsigned int)v80;
        v10 = v61;
        v11 = v65;
        v19 = v12;
      }
      v79[v32] = v31;
      v13 = v16;
      LODWORD(v80) = v80 + 1;
    }
    ++v12;
  }
  while ( v10 != v19 );
  v59 = v13;
  v68 = v11;
  if ( !v13 )
    goto LABEL_12;
  v6 = (unsigned int)v77;
  if ( (_DWORD)v77 && (unsigned __int8)sub_F8FD20((void **)&v76) )
  {
    v64 = &v76;
    v59 = v68;
    v68 = v13;
    goto LABEL_27;
  }
  v64 = &v79;
  if ( !(unsigned __int8)sub_F8FD20((void **)&v79) )
  {
LABEL_12:
    v17 = 0;
    goto LABEL_13;
  }
LABEL_27:
  v23 = sub_AD6890((*v64)[*((unsigned int *)v64 + 2) - 1], 0);
  v58 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v23 + 8), *((unsigned int *)v64 + 2), 0);
  v24 = **(_QWORD **)(a2 - 8);
  if ( !sub_AC30F0(v23) )
  {
    v41 = sub_BD5D20(v24);
    v42 = *(_QWORD *)(a3 + 80);
    v70[0] = v41;
    v71 = 773;
    v70[1] = v43;
    v70[2] = ".off";
    v44 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v42 + 32LL))(
            v42,
            13,
            v24,
            v23,
            0,
            0);
    if ( !v44 )
    {
      v75 = 257;
      v57 = sub_B504D0(13, v24, v23, (__int64)&v72, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v57,
        v70,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v47 = *(unsigned int **)a3;
      v44 = v57;
      v48 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v48 )
      {
        do
        {
          v49 = *((_QWORD *)v47 + 1);
          v50 = *v47;
          v47 += 4;
          sub_B99FD0(v57, v50, v49);
        }
        while ( (unsigned int *)v48 != v47 );
        v44 = v57;
      }
    }
    v24 = v44;
  }
  if ( sub_AC30F0((__int64)v58) && *((_DWORD *)v64 + 2) )
  {
    v46 = (__int64 *)sub_BD5C60(a2);
    v6 = sub_ACD6D0(v46);
  }
  else
  {
    v72 = "switch";
    v75 = 259;
    v6 = sub_92B530((unsigned int **)a3, 0x24u, v24, v58, (__int64)&v72);
  }
  v25 = sub_F94450((__int64 *)a3, v6, v59, v68, 0, 0);
  if ( (unsigned __int8)sub_BC8700(a2) )
  {
    v6 = (unsigned __int64)&v72;
    v73 = 0x800000000LL;
    v72 = v74;
    sub_F8F540((_BYTE *)a2, (__int64)&v72);
    v45 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
    if ( v45 == (_DWORD)v73 )
    {
      v51 = v45;
      if ( v45 )
      {
        v52 = v72;
        v53 = 1;
        v6 = 0;
        v54 = &v72[8 * v51];
        v55 = 0;
        do
        {
          v56 = *(_QWORD *)v52;
          if ( v59 == *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * v53) )
            v6 += v56;
          else
            v55 += v56;
          v52 += 8;
          v53 += 2;
        }
        while ( v54 != v52 );
        if ( v6 > 0xFFFFFFFF || v55 > 0xFFFFFFFF )
        {
          do
          {
            do
            {
              v6 >>= 1;
              v55 >>= 1;
            }
            while ( v6 > 0xFFFFFFFF );
          }
          while ( v55 > 0xFFFFFFFF );
        }
      }
      else
      {
        LODWORD(v55) = 0;
        v6 = 0;
      }
      sub_F8EA30(v25, v6, v55);
    }
    if ( v72 != v74 )
      _libc_free(v72, v6);
  }
  for ( i = *(_QWORD *)(v59 + 56); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) != 84 )
      break;
    v27 = *((_DWORD *)v64 + 2);
    v28 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
    if ( v59 != v28 || !v28 )
      --v27;
    v29 = 0;
    if ( v27 )
    {
      do
      {
        if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
        {
          v30 = 0;
          while ( 1 )
          {
            v6 = (unsigned int)v30;
            if ( *(_QWORD *)(a2 + 40) == *(_QWORD *)(*(_QWORD *)(i - 32) + 32LL * *(unsigned int *)(i + 48) + 8 * v30) )
              break;
            if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) == (_DWORD)++v30 )
              goto LABEL_49;
          }
        }
        else
        {
LABEL_49:
          v6 = 0xFFFFFFFFLL;
        }
        ++v29;
        sub_B48BF0(i - 24, v6, 1);
      }
      while ( v29 != v27 );
    }
  }
  for ( j = *(_QWORD *)(v68 + 56); ; j = *(_QWORD *)(j + 8) )
  {
    if ( !j )
      BUG();
    if ( *(_BYTE *)(j - 24) != 84 )
      break;
    v35 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1 - *((_DWORD *)v64 + 2);
    v36 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
    if ( !v36 || v68 != v36 )
      --v35;
    v37 = 0;
    if ( v35 )
    {
      do
      {
        if ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) != 0 )
        {
          v38 = 0;
          while ( 1 )
          {
            v6 = (unsigned int)v38;
            if ( *(_QWORD *)(a2 + 40) == *(_QWORD *)(*(_QWORD *)(j - 32) + 32LL * *(unsigned int *)(j + 48) + 8 * v38) )
              break;
            if ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) == (_DWORD)++v38 )
              goto LABEL_63;
          }
        }
        else
        {
LABEL_63:
          v6 = 0xFFFFFFFFLL;
        }
        ++v37;
        sub_B48BF0(j - 24, v6, 1);
      }
      while ( v37 != v35 );
    }
  }
  if ( v62 == 36 )
  {
    v17 = 1;
    v6 = *(_QWORD *)(a1 + 8);
    sub_F9A0D0(a2, v6, 1);
    v39 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
    sub_B43D60((_QWORD *)a2);
    v40 = *(_QWORD *)(a1 + 8);
    if ( v40 )
    {
      v6 = (unsigned __int64)&v72;
      v73 = v39 | 4;
      v72 = v63;
      sub_FFB3D0(v40, &v72, 1);
    }
  }
  else
  {
    v17 = 1;
    sub_B43D60((_QWORD *)a2);
  }
LABEL_13:
  if ( v79 != (_QWORD *)v81 )
    _libc_free(v79, v6);
  if ( v76 != (_QWORD *)v78 )
    _libc_free(v76, v6);
  return v17;
}
