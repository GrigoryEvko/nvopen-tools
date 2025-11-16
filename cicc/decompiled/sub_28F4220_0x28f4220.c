// Function: sub_28F4220
// Address: 0x28f4220
//
__int64 __fastcall sub_28F4220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned int v7; // r14d
  char *v8; // rdx
  unsigned int v9; // r13d
  __int64 v10; // rbx
  unsigned int v11; // r12d
  char *v12; // rax
  int v13; // esi
  __int64 v14; // r15
  __int64 v15; // r15
  __int64 *v16; // r11
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // r10
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rcx
  _BYTE *v24; // rax
  char *v25; // rsi
  char *v26; // rbx
  int v27; // ecx
  char *v28; // rdi
  __int64 v29; // rbx
  __int64 v30; // rbx
  char *v31; // r12
  char *v32; // rbx
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 v35; // r13
  unsigned __int64 v36; // rdx
  int v37; // eax
  _BYTE *v38; // rdi
  __int64 v39; // r12
  char *v41; // rax
  char *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v51; // [rsp+10h] [rbp-D0h]
  __int64 *v52; // [rsp+18h] [rbp-C8h]
  _QWORD *v53; // [rsp+18h] [rbp-C8h]
  __int64 v54; // [rsp+18h] [rbp-C8h]
  __int64 v56; // [rsp+28h] [rbp-B8h]
  _QWORD v57[2]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v58; // [rsp+40h] [rbp-A0h]
  _QWORD *v59; // [rsp+50h] [rbp-90h] BYREF
  __int64 v60; // [rsp+58h] [rbp-88h]
  _BYTE v61[32]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v62; // [rsp+80h] [rbp-60h] BYREF
  __int64 v63; // [rsp+88h] [rbp-58h]
  _QWORD v64[10]; // [rsp+90h] [rbp-50h] BYREF

  v6 = a3;
  v7 = *(_DWORD *)(a3 + 8);
  v59 = v61;
  v8 = *(char **)a3;
  v60 = 0x400000000LL;
  if ( v7 > 1 )
  {
    v9 = 1;
    LODWORD(v10) = 0;
    v11 = v7;
    a6 = v6;
    while ( 1 )
    {
      v12 = &v8[16 * v9];
      v13 = *((_DWORD *)v12 + 2);
      if ( !v13 )
        break;
      v14 = (unsigned int)v10;
      LODWORD(v10) = v9;
      v15 = 16 * v14;
      if ( v13 == *(_DWORD *)&v8[v15 + 8] )
      {
        v63 = 0x400000000LL;
        v10 = v9 + 1;
        v16 = (__int64 *)&v62;
        v17 = 16 * v10;
        v62 = v64;
        v18 = *(_QWORD *)&v8[v15];
        LODWORD(v63) = 1;
        v64[0] = v18;
        v19 = *(_QWORD *)v12;
        v20 = v64;
        v21 = 1;
        v22 = v19;
        while ( 1 )
        {
          v20[v21] = v22;
          v21 = (unsigned int)(v63 + 1);
          LODWORD(v63) = v63 + 1;
          if ( v11 <= (unsigned int)v10 )
            break;
          v23 = *(_QWORD *)a6 + v17;
          if ( *(_DWORD *)(v23 + 8) != *(_DWORD *)(*(_QWORD *)a6 + v15 + 8) )
            break;
          v22 = *(_QWORD *)v23;
          if ( v21 + 1 > (unsigned __int64)HIDWORD(v63) )
          {
            v51 = a6;
            v52 = v16;
            sub_C8D5F0((__int64)v16, v64, v21 + 1, 8u, a5, a6);
            v21 = (unsigned int)v63;
            a6 = v51;
            v16 = v52;
          }
          v20 = v62;
          v17 += 16;
          LODWORD(v10) = v10 + 1;
        }
        v53 = (_QWORD *)a6;
        v24 = (_BYTE *)sub_28EA500(a2, v16);
        a6 = (__int64)v53;
        *(_QWORD *)(*v53 + v15) = v24;
        if ( *v24 > 0x1Cu )
        {
          v58 = v24;
          v57[0] = 0;
          v57[1] = 0;
          if ( v24 != (_BYTE *)-4096LL && v24 != (_BYTE *)-8192LL )
          {
            sub_BD73F0((__int64)v57);
            a6 = (__int64)v53;
          }
          v54 = a6;
          sub_28F19A0(a1 + 64, v57);
          a6 = v54;
          if ( v58 != 0 && v58 + 4096 != 0 && v58 != (_BYTE *)-8192LL )
          {
            sub_BD60C0(v57);
            a6 = v54;
          }
        }
        if ( v62 != v64 )
        {
          v56 = a6;
          _libc_free((unsigned __int64)v62);
          a6 = v56;
        }
        v9 = v10 + 1;
        v8 = *(char **)a6;
        if ( v11 <= (int)v10 + 1 )
          break;
      }
      else if ( v11 <= ++v9 )
      {
        break;
      }
    }
    v7 = *(_DWORD *)(a6 + 8);
    v6 = a6;
  }
  v25 = &v8[16 * v7];
  if ( v25 == v8 )
  {
    *(_DWORD *)(v6 + 8) = 0;
  }
  else
  {
    v26 = v8;
    while ( 1 )
    {
      v28 = v26;
      v26 += 16;
      if ( v25 == v26 )
        break;
      v27 = *((_DWORD *)v26 - 2);
      if ( v27 == *((_DWORD *)v26 + 2) )
      {
        if ( v25 == v28 )
        {
          v26 = &v8[16 * v7];
        }
        else
        {
          v41 = v28 + 32;
          if ( v25 != v28 + 32 )
          {
            while ( 1 )
            {
              if ( *((_DWORD *)v41 + 2) != v27 )
              {
                v28 += 16;
                *(_QWORD *)v28 = *(_QWORD *)v41;
                *((_DWORD *)v28 + 2) = *((_DWORD *)v41 + 2);
              }
              v41 += 16;
              if ( v25 == v41 )
                break;
              v27 = *((_DWORD *)v28 + 2);
            }
            v8 = *(char **)v6;
            v42 = v28 + 16;
            a5 = *(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8) - (_QWORD)v25;
            v26 = &v42[a5];
            if ( v25 != (char *)(*(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8)) )
            {
              memmove(v42, v25, *(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8) - (_QWORD)v25);
              v8 = *(char **)v6;
            }
          }
        }
        break;
      }
    }
    v29 = (v26 - v8) >> 4;
    *(_DWORD *)(v6 + 8) = v29;
    v30 = 16LL * (unsigned int)v29;
    v31 = &v8[v30];
    if ( &v8[v30] != v8 )
    {
      v32 = v8;
      do
      {
        while ( 1 )
        {
          v33 = *((_DWORD *)v32 + 2);
          if ( (v33 & 1) != 0 )
            break;
          v32 += 16;
          *((_DWORD *)v32 - 2) = v33 >> 1;
          if ( v31 == v32 )
            goto LABEL_34;
        }
        v34 = (unsigned int)v60;
        v35 = *(_QWORD *)v32;
        v36 = (unsigned int)v60 + 1LL;
        if ( v36 > HIDWORD(v60) )
        {
          sub_C8D5F0((__int64)&v59, v61, v36, 8u, a5, a6);
          v34 = (unsigned int)v60;
        }
        v32 += 16;
        v59[v34] = v35;
        LODWORD(v60) = v60 + 1;
        *((_DWORD *)v32 - 2) >>= 1;
      }
      while ( v31 != v32 );
LABEL_34:
      v8 = *(char **)v6;
    }
  }
  if ( *((_DWORD *)v8 + 2) )
  {
    v44 = sub_28F4220(a1, a2, v6);
    v47 = (unsigned int)v60;
    v48 = (unsigned int)v60 + 1LL;
    if ( v48 > HIDWORD(v60) )
    {
      sub_C8D5F0((__int64)&v59, v61, v48, 8u, v45, v46);
      v47 = (unsigned int)v60;
    }
    v59[v47] = v44;
    LODWORD(v60) = v60 + 1;
    v49 = (unsigned int)v60;
    if ( (unsigned __int64)(unsigned int)v60 + 1 > HIDWORD(v60) )
    {
      sub_C8D5F0((__int64)&v59, v61, (unsigned int)v60 + 1LL, 8u, v45, v46);
      v49 = (unsigned int)v60;
    }
    v59[v49] = v44;
    v37 = v60 + 1;
    LODWORD(v60) = v60 + 1;
  }
  else
  {
    v37 = v60;
  }
  if ( v37 == 1 )
  {
    v38 = v59;
    v39 = *v59;
  }
  else
  {
    v43 = sub_28EA500(a2, (__int64 *)&v59);
    v38 = v59;
    v39 = v43;
  }
  if ( v38 != v61 )
    _libc_free((unsigned __int64)v38);
  return v39;
}
