// Function: sub_320D160
// Address: 0x320d160
//
unsigned int *__fastcall sub_320D160(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  unsigned int *result; // rax
  __int64 v8; // rdx
  unsigned int *v9; // r13
  unsigned int *v10; // r15
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  __int64 v13; // rdi
  __int64 v14; // rcx
  unsigned int v15; // r8d
  __int64 v16; // r11
  int v17; // r12d
  __int64 *v18; // rbx
  unsigned int i; // eax
  __int64 *v20; // rdx
  __int64 v21; // r9
  __int64 v22; // rdx
  char *v23; // rbx
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rdi
  _QWORD *v27; // rdx
  bool v28; // zf
  int v29; // ebx
  __int16 v30; // ax
  _QWORD *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r8
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // r13
  __int64 v39; // r15
  __int64 *v40; // rdx
  __int64 v41; // rbx
  int v42; // esi
  int v43; // esi
  __int64 *v44; // rdi
  unsigned int v45; // eax
  int v46; // eax
  __int64 v47; // [rsp+8h] [rbp-108h]
  __int64 v48; // [rsp+10h] [rbp-100h]
  unsigned int *v49; // [rsp+18h] [rbp-F8h]
  unsigned int *v50; // [rsp+20h] [rbp-F0h]
  char v51; // [rsp+37h] [rbp-D9h]
  __int64 v52; // [rsp+38h] [rbp-D8h]
  __int64 v53; // [rsp+40h] [rbp-D0h]
  _QWORD *v55; // [rsp+58h] [rbp-B8h]
  unsigned int v56; // [rsp+6Ch] [rbp-A4h] BYREF
  __int64 v57; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v58; // [rsp+78h] [rbp-98h] BYREF
  __int64 v59; // [rsp+80h] [rbp-90h] BYREF
  __int64 v60; // [rsp+88h] [rbp-88h] BYREF
  __int64 v61; // [rsp+90h] [rbp-80h]
  __int64 v62; // [rsp+98h] [rbp-78h]
  unsigned int v63; // [rsp+A0h] [rbp-70h]
  char *v64; // [rsp+A8h] [rbp-68h]
  __int64 v65; // [rsp+B0h] [rbp-60h]
  char v66; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 v67; // [rsp+C0h] [rbp-50h]
  unsigned int v68; // [rsp+C8h] [rbp-48h]
  char v69; // [rsp+D0h] [rbp-40h]

  v52 = 0;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
  v4 = *(__int64 **)(v3 + 16);
  v5 = *v4;
  v6 = *(__int64 (**)(void))(*v4 + 136);
  if ( v6 != sub_2DD19D0 )
  {
    v44 = *(__int64 **)(v3 + 16);
    v52 = v6();
    v5 = *v44;
  }
  v48 = (*(__int64 (**)(void))(v5 + 200))();
  result = *(unsigned int **)(v3 + 752);
  v8 = 8LL * *(unsigned int *)(v3 + 760);
  v9 = &result[v8];
  if ( &result[v8] != result )
  {
    while ( 1 )
    {
      v10 = result;
      if ( !*((_BYTE *)result + 4) )
        break;
      result += 8;
      if ( v9 == result )
        return result;
    }
    while ( 1 )
    {
      if ( v9 == v10 )
        return result;
      if ( !*((_QWORD *)v10 + 1) )
        goto LABEL_35;
      v11 = *((_QWORD *)v10 + 3);
      v12 = *(_BYTE *)(v11 - 16);
      if ( (v12 & 2) != 0 )
      {
        if ( *(_DWORD *)(v11 - 24) != 2 )
          goto LABEL_11;
        v22 = *(_QWORD *)(v11 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v11 - 16) >> 6) & 0xF) != 2 )
        {
LABEL_11:
          v13 = 0;
          goto LABEL_12;
        }
        v22 = v11 - 16 - 8LL * ((v12 >> 2) & 0xF);
      }
      v13 = *(_QWORD *)(v22 + 8);
LABEL_12:
      v14 = *((_QWORD *)v10 + 1);
      v60 = v13;
      v15 = *(_DWORD *)(a2 + 24);
      v59 = v14;
      if ( v15 )
      {
        v16 = *(_QWORD *)(a2 + 8);
        v17 = 1;
        v18 = 0;
        for ( i = (v15 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                    | ((unsigned __int64)(((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; i = (v15 - 1) & v45 )
        {
          v20 = (__int64 *)(v16 + 16LL * i);
          v21 = *v20;
          if ( v14 == *v20 && v13 == v20[1] )
            break;
          if ( v21 == -4096 )
          {
            if ( v20[1] == -4096 )
            {
              if ( v18 )
                v20 = v18;
              ++*(_QWORD *)a2;
              v46 = *(_DWORD *)(a2 + 16);
              v58 = v20;
              v43 = v46 + 1;
              if ( 4 * (v46 + 1) >= 3 * v15 )
                goto LABEL_68;
              v41 = a2;
              if ( v15 - *(_DWORD *)(a2 + 20) - v43 > v15 >> 3 )
                goto LABEL_70;
              v42 = v15;
              goto LABEL_69;
            }
          }
          else if ( v21 == -8192 && v20[1] == -8192 && !v18 )
          {
            v18 = (__int64 *)(v16 + 16LL * i);
          }
          v45 = v17 + i;
          ++v17;
        }
      }
      else
      {
        v58 = 0;
        ++*(_QWORD *)a2;
LABEL_68:
        v41 = a2;
        v42 = 2 * v15;
LABEL_69:
        sub_3201030(v41, v42);
        sub_31FC520(v41, &v59, &v58);
        v14 = v59;
        v20 = v58;
        v43 = *(_DWORD *)(v41 + 16) + 1;
LABEL_70:
        *(_DWORD *)(a2 + 16) = v43;
        if ( *v20 != -4096 || v20[1] != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v20 = v14;
        v20[1] = v60;
        v11 = *((_QWORD *)v10 + 3);
      }
      v53 = sub_35051D0(a1 + 72, v11);
      if ( v53 )
      {
        v57 = 0;
        v26 = *((_QWORD *)v10 + 2);
        if ( !v26 )
          goto LABEL_48;
        v27 = *(_QWORD **)(v26 + 16);
        if ( (unsigned int)((__int64)(*(_QWORD *)(v26 + 24) - (_QWORD)v27) >> 3) == 1 && *v27 == 6 )
        {
          v51 = 1;
        }
        else
        {
          if ( !(unsigned __int8)sub_AF4AF0(v26, &v57) )
            goto LABEL_35;
LABEL_48:
          v51 = 0;
        }
        v28 = *((_BYTE *)v10 + 4) == 0;
        v56 = 0;
        if ( !v28 )
          abort();
        v29 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, unsigned int *))(*(_QWORD *)v52 + 224LL))(
                v52,
                *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL),
                *v10,
                &v56);
        v30 = sub_E91F10(v48, v56);
        v60 = 0;
        HIWORD(v58) = v30;
        LODWORD(v58) = 2 * (v57 + v29) + 1;
        WORD2(v58) = 0;
        v61 = 0;
        v62 = 0;
        v63 = 0;
        v64 = &v66;
        v65 = 0;
        v66 = 0;
        v69 = 0;
        v59 = *((_QWORD *)v10 + 1);
        v31 = *(_QWORD **)(v53 + 80);
        v55 = &v31[2 * *(unsigned int *)(v53 + 88)];
        if ( v55 != v31 )
        {
          v50 = v9;
          v49 = v10;
          do
          {
            v38 = sub_3211F40(a1, *v31);
            v39 = sub_3211FB0(a1, v31[1]);
            if ( !v39 )
              v39 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 400LL);
            v32 = sub_3202A70((__int64)&v60, (__int64 *)&v58);
            v34 = *(unsigned int *)(v32 + 8);
            v35 = *(unsigned int *)(v32 + 12);
            v36 = *(_DWORD *)(v32 + 8);
            if ( v34 >= v35 )
            {
              if ( v35 < v34 + 1 )
              {
                v47 = v32;
                sub_C8D5F0(v32, (const void *)(v32 + 16), v34 + 1, 0x10u, v33, v34 + 1);
                v32 = v47;
                v34 = *(unsigned int *)(v47 + 8);
              }
              v40 = (__int64 *)(*(_QWORD *)v32 + 16 * v34);
              *v40 = v38;
              v40[1] = v39;
              ++*(_DWORD *)(v32 + 8);
            }
            else
            {
              v37 = (__int64 *)(*(_QWORD *)v32 + 16 * v34);
              if ( v37 )
              {
                *v37 = v38;
                v37[1] = v39;
                v36 = *(_DWORD *)(v32 + 8);
              }
              *(_DWORD *)(v32 + 8) = v36 + 1;
            }
            v31 += 2;
          }
          while ( v55 != v31 );
          v9 = v50;
          v10 = v49;
        }
        if ( v51 )
          v66 = 1;
        sub_320CF50(a1, &v59, v53);
        if ( v69 )
        {
          v69 = 0;
          if ( v68 > 0x40 )
          {
            if ( v67 )
              j_j___libc_free_0_0(v67);
          }
        }
        v23 = v64;
        v24 = (unsigned __int64)&v64[40 * (unsigned int)v65];
        if ( v64 != (char *)v24 )
        {
          do
          {
            v24 -= 40LL;
            v25 = *(_QWORD *)(v24 + 8);
            if ( v25 != v24 + 24 )
              _libc_free(v25);
          }
          while ( v23 != (char *)v24 );
          v24 = (unsigned __int64)v64;
        }
        if ( (char *)v24 != &v66 )
          _libc_free(v24);
        sub_C7D6A0(v61, 12LL * v63, 4);
      }
LABEL_35:
      result = v10 + 8;
      if ( v9 == v10 + 8 )
        return result;
      while ( 1 )
      {
        v10 = result;
        if ( !*((_BYTE *)result + 4) )
          break;
        result += 8;
        if ( v9 == result )
          return result;
      }
    }
  }
  return result;
}
