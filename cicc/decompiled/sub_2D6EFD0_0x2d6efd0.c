// Function: sub_2D6EFD0
// Address: 0x2d6efd0
//
void __fastcall sub_2D6EFD0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *i; // rbx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  bool v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // r14
  __int64 v20; // r9
  int v21; // edi
  int v22; // r14d
  __int64 v23; // r8
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // rdi
  unsigned int v27; // esi
  _QWORD *v28; // rcx
  __int64 v29; // r10
  int v30; // edx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 v36; // rdi
  int v37; // eax
  int v38; // edx
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-A8h]
  __int64 v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+8h] [rbp-A8h]
  int v43; // [rsp+8h] [rbp-A8h]
  _QWORD *v44; // [rsp+18h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  _QWORD *v46; // [rsp+18h] [rbp-98h]
  _QWORD *v48; // [rsp+28h] [rbp-88h]
  _QWORD *v49; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v50[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v51; // [rsp+50h] [rbp-60h]
  unsigned __int64 v52; // [rsp+60h] [rbp-50h] BYREF
  __int64 v53; // [rsp+68h] [rbp-48h]
  __int64 v54; // [rsp+70h] [rbp-40h]
  __int64 v55; // [rsp+78h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 4 )
  {
    for ( i = a1 + 8; ; i += 4 )
    {
      v6 = a1[2];
      v7 = *(i - 2);
      v48 = i;
      v8 = a1[3];
      if ( v6 == v7 )
        goto LABEL_35;
      if ( v8 != *(i - 1) )
      {
        v9 = v8 > *(i - 1);
        goto LABEL_7;
      }
      v54 = *(i - 2);
      v45 = a3 + 728;
      v52 = 0;
      v53 = 0;
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
      {
        v40 = v6;
        sub_BD73F0((__int64)&v52);
        v6 = v40;
      }
      v18 = *(_DWORD *)(a3 + 752);
      if ( !v18 )
        break;
      v20 = v54;
      v23 = *(_QWORD *)(a3 + 736);
      v24 = (v18 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v25 = v23 + 32 * v24;
      v26 = *(_QWORD *)(v25 + 16);
      if ( v54 == v26 )
      {
LABEL_47:
        v22 = *(_DWORD *)(v25 + 24);
        goto LABEL_48;
      }
      v38 = 1;
      v19 = 0;
      while ( v26 != -4096 )
      {
        if ( v26 == -8192 && !v19 )
          v19 = v25;
        LODWORD(v24) = (v18 - 1) & (v38 + v24);
        v25 = v23 + 32LL * (unsigned int)v24;
        v26 = *(_QWORD *)(v25 + 16);
        if ( v54 == v26 )
          goto LABEL_47;
        ++v38;
      }
      if ( !v19 )
        v19 = v25;
      v39 = *(_DWORD *)(a3 + 744);
      ++*(_QWORD *)(a3 + 728);
      v21 = v39 + 1;
      v50[0] = v19;
      if ( 4 * (v39 + 1) >= 3 * v18 )
        goto LABEL_41;
      if ( v18 - *(_DWORD *)(a3 + 748) - v21 > v18 >> 3 )
        goto LABEL_43;
      v41 = v6;
LABEL_42:
      sub_2D6E640(v45, v18);
      sub_2D67BB0(v45, (__int64)&v52, v50);
      v19 = v50[0];
      v20 = v54;
      v6 = v41;
      v21 = *(_DWORD *)(a3 + 744) + 1;
LABEL_43:
      *(_DWORD *)(a3 + 744) = v21;
      if ( *(_QWORD *)(v19 + 16) != -4096 )
        --*(_DWORD *)(a3 + 748);
      v42 = v6;
      sub_2D57220((_QWORD *)v19, v20);
      *(_DWORD *)(v19 + 24) = 0;
      v22 = 0;
      v6 = v42;
LABEL_48:
      v50[0] = 0;
      v50[1] = 0;
      v51 = v6;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD73F0((__int64)v50);
      v27 = *(_DWORD *)(a3 + 752);
      if ( !v27 )
      {
        ++*(_QWORD *)(a3 + 728);
        v49 = 0;
        goto LABEL_53;
      }
      v29 = v51;
      v33 = *(_QWORD *)(a3 + 736);
      v32 = v51;
      v34 = (v27 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v35 = v33 + 32 * v34;
      v36 = *(_QWORD *)(v35 + 16);
      if ( v51 != v36 )
      {
        v43 = 1;
        v28 = 0;
        while ( v36 != -4096 )
        {
          if ( v36 == -8192 && !v28 )
            v28 = (_QWORD *)v35;
          LODWORD(v34) = (v27 - 1) & (v43 + v34);
          v35 = v33 + 32LL * (unsigned int)v34;
          v36 = *(_QWORD *)(v35 + 16);
          if ( v51 == v36 )
          {
            v32 = v51;
            goto LABEL_59;
          }
          ++v43;
        }
        v37 = *(_DWORD *)(a3 + 744);
        if ( !v28 )
          v28 = (_QWORD *)v35;
        ++*(_QWORD *)(a3 + 728);
        v30 = v37 + 1;
        v49 = v28;
        if ( 4 * (v37 + 1) >= 3 * v27 )
        {
LABEL_53:
          v27 *= 2;
        }
        else if ( v27 - *(_DWORD *)(a3 + 748) - v30 > v27 >> 3 )
        {
          goto LABEL_55;
        }
        sub_2D6E640(v45, v27);
        sub_2D67BB0(v45, (__int64)v50, &v49);
        v28 = v49;
        v29 = v51;
        v30 = *(_DWORD *)(a3 + 744) + 1;
LABEL_55:
        *(_DWORD *)(a3 + 744) = v30;
        if ( v28[2] != -4096 )
          --*(_DWORD *)(a3 + 748);
        v46 = v28;
        sub_2D57220(v28, v29);
        v31 = 0;
        *((_DWORD *)v46 + 6) = 0;
        v32 = v51;
        goto LABEL_60;
      }
LABEL_59:
      v31 = *(_DWORD *)(v35 + 24);
LABEL_60:
      v9 = v22 < v31;
      if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
        sub_BD60C0(v50);
      if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
        sub_BD60C0(&v52);
LABEL_7:
      if ( v9 )
      {
        v10 = *(i - 2);
        v52 = 0;
        v53 = 0;
        v54 = v10;
        if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
          sub_BD6050(&v52, *(i - 4) & 0xFFFFFFFFFFFFFFF8LL);
        v11 = i;
        v12 = (char *)(i - 4) - (char *)a1;
        v55 = *(i - 1);
        if ( v12 > 0 )
        {
          v44 = i;
          v13 = v12 >> 5;
          do
          {
            v14 = *(v11 - 6);
            v15 = *(v11 - 2);
            v11 -= 4;
            if ( v14 != v15 )
            {
              if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
                sub_BD60C0(v11);
              v11[2] = v14;
              if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                sub_BD73F0((__int64)v11);
            }
            v11[3] = *(v11 - 1);
            --v13;
          }
          while ( v13 );
          i = v44;
        }
        v16 = v54;
        v17 = a1[2];
        if ( v54 != v17 )
        {
          if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
            sub_BD60C0(a1);
          a1[2] = v16;
          if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
            sub_BD73F0((__int64)a1);
          v17 = v54;
        }
        a1[3] = v55;
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
          sub_BD60C0(&v52);
        goto LABEL_33;
      }
LABEL_35:
      sub_2D6EA50(i - 4, a3);
LABEL_33:
      if ( a2 == v48 )
        return;
    }
    ++*(_QWORD *)(a3 + 728);
    v50[0] = 0;
LABEL_41:
    v41 = v6;
    v18 *= 2;
    goto LABEL_42;
  }
}
