// Function: sub_2C71E10
// Address: 0x2c71e10
//
unsigned __int64 __fastcall sub_2C71E10(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v6; // rsi
  __int64 v7; // rcx
  _BYTE *v8; // r15
  char *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // rax
  char *v15; // r13
  signed __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 result; // rax
  __int64 v19; // rcx
  unsigned int v20; // r12d
  __int64 v21; // rdx
  _QWORD *v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // rbx
  __int64 v25; // r14
  _BYTE *v26; // rsi
  _QWORD *v27; // rdx
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // r14d
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // rdx
  __int64 v35; // r9
  unsigned int v36; // eax
  __int64 *v37; // rdi
  __int64 v38; // rcx
  int v39; // eax
  __int64 v40; // r10
  __int64 v41; // rcx
  __int64 *v42; // r14
  int v43; // eax
  __int64 *v44; // rdi
  int v45; // eax
  int v46; // eax
  __int64 v47; // r10
  __int64 *v48; // rdi
  __int64 v49; // rsi
  unsigned int v50; // r11d
  __int64 *v51; // rbx
  unsigned int v52; // r11d
  unsigned int v53; // r11d
  __int64 v54; // [rsp+8h] [rbp-118h]
  __int64 v55; // [rsp+10h] [rbp-110h]
  unsigned int v56; // [rsp+1Ch] [rbp-104h]
  __int64 v57; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+28h] [rbp-F8h]
  __int64 v59; // [rsp+30h] [rbp-F0h]
  unsigned int v60; // [rsp+30h] [rbp-F0h]
  void *v61; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v62; // [rsp+50h] [rbp-D0h]
  _BYTE s[48]; // [rsp+58h] [rbp-C8h] BYREF
  int v64; // [rsp+88h] [rbp-98h]
  __int64 v65; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v66; // [rsp+98h] [rbp-88h]
  char *v67; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-78h]
  _BYTE v69[48]; // [rsp+B0h] [rbp-70h] BYREF
  int v70; // [rsp+E0h] [rbp-40h]

  v6 = (char *)a1[7];
  v7 = *a1;
  v58 = (__int64)(a1 + 5);
  v8 = (_BYTE *)a1[5];
  v9 = (char *)a1[6];
  v10 = *(unsigned int *)(*a1 + 32LL);
  if ( v10 > (v6 - v8) >> 3 )
  {
    v11 = v9 - v8;
    v12 = 8 * v10;
    v13 = v11;
    if ( *(_DWORD *)(v7 + 32) )
    {
      v14 = sub_22077B0(8 * v10);
      v8 = (_BYTE *)a1[5];
      v15 = (char *)v14;
      v16 = a1[6] - (_QWORD)v8;
      if ( v16 <= 0 )
        goto LABEL_4;
    }
    else
    {
      v16 = v11;
      v15 = 0;
      if ( v11 <= 0 )
      {
LABEL_4:
        if ( !v8 )
        {
LABEL_5:
          v9 = &v15[v13];
          v6 = &v15[v12];
          a1[5] = v15;
          v7 = *a1;
          a1[6] = &v15[v13];
          a1[7] = &v15[v12];
          goto LABEL_6;
        }
LABEL_57:
        j_j___libc_free_0((unsigned __int64)v8);
        goto LABEL_5;
      }
    }
    memmove(v15, v8, v16);
    goto LABEL_57;
  }
LABEL_6:
  v17 = *(_QWORD *)(v7 + 96);
  v65 = v17;
  if ( v6 == v9 )
  {
    sub_2C6F5C0(v58, v6, &v65);
    result = a1[6];
  }
  else
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = v17;
      v9 = (char *)a1[6];
    }
    result = (unsigned __int64)(v9 + 8);
    a1[6] = result;
  }
  v19 = a1[5];
  if ( result != v19 )
  {
    v20 = 0;
    v21 = 0;
    v22 = a1;
    v54 = (__int64)(a1 + 1);
    do
    {
      v59 = 8 * v21;
      v23 = *(_QWORD *)(v19 + 8 * v21);
      v24 = *(_QWORD **)(v23 + 24);
      v25 = *(unsigned int *)(v23 + 32);
      if ( *(_DWORD *)(v23 + 32) )
      {
        do
        {
          while ( 1 )
          {
            v26 = (_BYTE *)v22[6];
            if ( v26 != (_BYTE *)v22[7] )
              break;
            v27 = v24++;
            sub_2C71C80(v58, v26, v27);
            if ( !--v25 )
              goto LABEL_19;
          }
          if ( v26 )
          {
            *(_QWORD *)v26 = *v24;
            v26 = (_BYTE *)v22[6];
          }
          ++v24;
          v22[6] = v26 + 8;
          --v25;
        }
        while ( v25 );
LABEL_19:
        v19 = v22[5];
      }
      v28 = **(_QWORD **)(v19 + v59);
      v29 = *v22;
      v61 = s;
      v30 = *(unsigned int *)(v29 + 32);
      v62 = 0x600000000LL;
      v31 = (unsigned int)(v30 + 63) >> 6;
      v32 = v31;
      if ( v31 > 6 )
      {
        v55 = v28;
        v56 = v30;
        sub_C8D5F0((__int64)&v61, s, v31, 8u, a5, v30);
        memset(v61, 0, 8LL * v31);
        v30 = v56;
        LODWORD(v62) = v31;
        v66 = v20;
        v64 = v56;
        v67 = v69;
        v65 = v55;
        v68 = 0x600000000LL;
      }
      else
      {
        if ( v31 )
        {
          v32 = 8LL * v31;
          if ( v32 )
          {
            v57 = v28;
            v60 = v30;
            memset(s, 0, v32);
            v28 = v57;
            v30 = v60;
          }
        }
        LODWORD(v62) = v31;
        v64 = v30;
        v67 = v69;
        v65 = v28;
        v66 = v20;
        v68 = 0x600000000LL;
        if ( !v31 )
        {
          v33 = *((_DWORD *)v22 + 8);
          v70 = v30;
          if ( !v33 )
            goto LABEL_35;
          goto LABEL_26;
        }
      }
      sub_2C6DEF0((__int64)&v67, (__int64)&v61, v32, v19, a5, v30);
      v33 = *((_DWORD *)v22 + 8);
      v70 = v64;
      if ( !v33 )
      {
LABEL_35:
        ++v22[1];
        goto LABEL_36;
      }
LABEL_26:
      v34 = v65;
      v35 = v22[2];
      v36 = (v33 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
      v37 = (__int64 *)(v35 + 88LL * v36);
      v38 = *v37;
      if ( v65 != *v37 )
      {
        a5 = 1;
        v42 = 0;
        while ( v38 != -4096 )
        {
          if ( v42 || v38 != -8192 )
            v37 = v42;
          v50 = a5 + 1;
          v36 = (v33 - 1) & (a5 + v36);
          a5 = 5LL * v36;
          v51 = (__int64 *)(v35 + 88LL * v36);
          v38 = *v51;
          if ( v65 == *v51 )
            goto LABEL_27;
          v42 = v37;
          a5 = v50;
          v37 = (__int64 *)(v35 + 88LL * v36);
        }
        v45 = *((_DWORD *)v22 + 6);
        if ( !v42 )
          v42 = v37;
        ++v22[1];
        v43 = v45 + 1;
        if ( 4 * v43 >= 3 * v33 )
        {
LABEL_36:
          sub_2C6FB10(v54, 2 * v33);
          v39 = *((_DWORD *)v22 + 8);
          if ( !v39 )
            goto LABEL_81;
          a5 = (unsigned int)(v39 - 1);
          v40 = v22[2];
          v41 = (unsigned int)a5 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
          v42 = (__int64 *)(v40 + 88 * v41);
          v34 = *v42;
          v43 = *((_DWORD *)v22 + 6) + 1;
          if ( v65 != *v42 )
          {
            v35 = 1;
            v44 = 0;
            while ( v34 != -4096 )
            {
              if ( !v44 && v34 == -8192 )
                v44 = v42;
              v52 = v35 + 1;
              v41 = (unsigned int)a5 & ((_DWORD)v41 + (_DWORD)v35);
              v35 = 5LL * (unsigned int)v41;
              v42 = (__int64 *)(v40 + 88LL * (unsigned int)v41);
              v34 = *v42;
              if ( v65 == *v42 )
                goto LABEL_50;
              v35 = v52;
            }
            v34 = v65;
            if ( v44 )
              v42 = v44;
          }
        }
        else
        {
          v41 = v33 - *((_DWORD *)v22 + 7) - v43;
          if ( (unsigned int)v41 <= v33 >> 3 )
          {
            sub_2C6FB10(v54, v33);
            v46 = *((_DWORD *)v22 + 8);
            if ( !v46 )
            {
LABEL_81:
              ++*((_DWORD *)v22 + 6);
              BUG();
            }
            a5 = (unsigned int)(v46 - 1);
            v47 = v22[2];
            v35 = 1;
            v48 = 0;
            v34 = v65;
            v41 = (unsigned int)a5 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
            v42 = (__int64 *)(v47 + 88 * v41);
            v49 = *v42;
            v43 = *((_DWORD *)v22 + 6) + 1;
            if ( *v42 != v65 )
            {
              while ( v49 != -4096 )
              {
                if ( !v48 && v49 == -8192 )
                  v48 = v42;
                v53 = v35 + 1;
                v41 = (unsigned int)a5 & ((_DWORD)v41 + (_DWORD)v35);
                v35 = 5LL * (unsigned int)v41;
                v42 = (__int64 *)(v47 + 88LL * (unsigned int)v41);
                v49 = *v42;
                if ( v65 == *v42 )
                  goto LABEL_50;
                v35 = v53;
              }
              if ( v48 )
                v42 = v48;
            }
          }
        }
LABEL_50:
        *((_DWORD *)v22 + 6) = v43;
        if ( *v42 != -4096 )
          --*((_DWORD *)v22 + 7);
        *v42 = v34;
        *((_DWORD *)v42 + 2) = v66;
        v42[2] = (__int64)(v42 + 4);
        v42[3] = 0x600000000LL;
        if ( (_DWORD)v68 )
          sub_2C6DFD0((__int64)(v42 + 2), &v67, v34, v41, a5, v35);
        *((_DWORD *)v42 + 20) = v70;
      }
LABEL_27:
      if ( v67 != v69 )
        _libc_free((unsigned __int64)v67);
      if ( v61 != s )
        _libc_free((unsigned __int64)v61);
      v19 = v22[5];
      v21 = ++v20;
      result = (v22[6] - v19) >> 3;
    }
    while ( v20 < result );
  }
  return result;
}
