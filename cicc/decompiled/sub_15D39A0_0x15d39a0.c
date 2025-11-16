// Function: sub_15D39A0
// Address: 0x15d39a0
//
void __fastcall sub_15D39A0(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // edx
  _QWORD *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 *v12; // r12
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 i; // r15
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  int v30; // edx
  unsigned int v31; // eax
  __int64 *v32; // r14
  __int64 v33; // rdi
  __int64 v34; // r8
  __int64 v35; // rdi
  __int64 *v36; // r15
  unsigned int v37; // eax
  char *v38; // rdx
  __int64 *v39; // rax
  char **v40; // r12
  char **v41; // r15
  __int64 v42; // rax
  char *v43; // rsi
  unsigned int v44; // ecx
  char **v45; // rdi
  char *v46; // r10
  _QWORD *v47; // rbx
  _QWORD *v48; // r12
  unsigned __int64 v49; // rdi
  int v50; // edi
  int v51; // edx
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  unsigned __int64 v54; // rdi
  __int64 *v55; // [rsp+8h] [rbp-3F8h]
  __int64 v58; // [rsp+40h] [rbp-3C0h]
  unsigned int v59; // [rsp+4Ch] [rbp-3B4h]
  unsigned int v60; // [rsp+4Ch] [rbp-3B4h]
  __int64 v61; // [rsp+50h] [rbp-3B0h]
  __int64 v62; // [rsp+50h] [rbp-3B0h]
  __int64 *v63; // [rsp+50h] [rbp-3B0h]
  __int64 *v64; // [rsp+58h] [rbp-3A8h]
  __int64 v65; // [rsp+58h] [rbp-3A8h]
  int v66; // [rsp+58h] [rbp-3A8h]
  char *v67; // [rsp+68h] [rbp-398h] BYREF
  char *v68; // [rsp+70h] [rbp-390h] BYREF
  char *v69; // [rsp+78h] [rbp-388h] BYREF
  __int64 v70; // [rsp+80h] [rbp-380h]
  __int64 **v71; // [rsp+88h] [rbp-378h]
  __int64 v72; // [rsp+90h] [rbp-370h]
  _QWORD *v73; // [rsp+A0h] [rbp-360h] BYREF
  _QWORD *v74; // [rsp+A8h] [rbp-358h]
  _QWORD *v75; // [rsp+B0h] [rbp-350h]
  __int64 v76; // [rsp+B8h] [rbp-348h] BYREF
  _QWORD *v77; // [rsp+C0h] [rbp-340h]
  __int64 v78; // [rsp+C8h] [rbp-338h]
  unsigned int v79; // [rsp+D0h] [rbp-330h]
  __int64 v80; // [rsp+D8h] [rbp-328h]
  char **v81; // [rsp+E0h] [rbp-320h] BYREF
  int v82; // [rsp+E8h] [rbp-318h]
  char v83; // [rsp+F0h] [rbp-310h] BYREF
  __int64 *v84; // [rsp+130h] [rbp-2D0h] BYREF
  __int64 v85; // [rsp+138h] [rbp-2C8h]
  _BYTE v86[128]; // [rsp+140h] [rbp-2C0h] BYREF
  _QWORD *v87; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 v88; // [rsp+1C8h] [rbp-238h]
  _QWORD v89[70]; // [rsp+1D0h] [rbp-230h] BYREF

  v84 = (__int64 *)v86;
  v85 = 0x1000000000LL;
  v6 = *(_DWORD *)(a3 + 16);
  v72 = a1;
  LODWORD(v70) = v6;
  v71 = &v84;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v7 = (_QWORD *)sub_22077B0(8);
  *v7 = 0;
  v8 = *(_QWORD *)a3;
  v75 = v7 + 1;
  v74 = v7 + 1;
  v73 = v7;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = a2;
  v11 = sub_15D2530((__int64)&v73, v8, 0, 0, v9, v10, v70, (__int64)v71, v72);
  v12 = v84;
  v59 = v11;
  if ( v84 == &v84[(unsigned int)v85] )
  {
    v65 = a3;
  }
  else
  {
    v13 = *(_QWORD *)a3;
    v64 = &v84[(unsigned int)v85];
    v14 = a3;
    do
    {
      v15 = sub_15CC510(a1, *v12);
      v16 = *(_QWORD *)v15;
      v17 = v15;
      v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v15 + 56LL) + 80LL);
      if ( v18 )
        v18 -= 24;
      if ( v16 != v18 && v13 != v18 )
      {
        v61 = sub_15CC510(a1, v16);
        v19 = sub_15CC510(a1, v13);
        if ( v19 && (v18 = v61) != 0 )
        {
          while ( v19 != v18 )
          {
            if ( *(_DWORD *)(v18 + 16) < *(_DWORD *)(v19 + 16) )
            {
              v20 = v18;
              v18 = v19;
              v19 = v20;
            }
            v18 = *(_QWORD *)(v18 + 8);
            if ( !v18 )
              goto LABEL_13;
          }
          v18 = *(_QWORD *)v18;
        }
        else
        {
          v18 = 0;
        }
      }
LABEL_13:
      v21 = sub_15CC510(a1, v18);
      if ( v17 != v21 && *(_DWORD *)(v14 + 16) > *(_DWORD *)(v21 + 16) )
        v14 = v21;
      ++v12;
    }
    while ( v64 != v12 );
    v65 = v14;
  }
  if ( *(_QWORD *)(v65 + 8) )
  {
    if ( v59 )
    {
      for ( i = v59; ; --i )
      {
        v87 = (_QWORD *)sub_15CC510(a1, v73[i]);
        v23 = sub_15CBEB0(*(_QWORD **)(v87[1] + 24LL), *(_QWORD *)(v87[1] + 32LL), (__int64 *)&v87);
        v25 = *(_QWORD *)(v24 + 32);
        v26 = *v23;
        *v23 = *(_QWORD *)(v25 - 8);
        *(_QWORD *)(v25 - 8) = v26;
        *(_QWORD *)(v24 + 32) -= 8LL;
        v27 = *(_DWORD *)(a1 + 48);
        if ( v27 )
        {
          v28 = *(_QWORD *)(a1 + 32);
          v29 = 1;
          v30 = v27 - 1;
          v31 = (v27 - 1) & (((unsigned int)*v87 >> 9) ^ ((unsigned int)*v87 >> 4));
          v32 = (__int64 *)(v28 + 16LL * v31);
          v33 = *v32;
          if ( *v87 == *v32 )
          {
LABEL_24:
            v34 = v32[1];
            if ( v34 )
            {
              v35 = *(_QWORD *)(v34 + 24);
              if ( v35 )
              {
                v62 = v32[1];
                j_j___libc_free_0(v35, *(_QWORD *)(v34 + 40) - v35);
                v34 = v62;
              }
              j_j___libc_free_0(v34, 56);
            }
            *v32 = -16;
            --*(_DWORD *)(a1 + 40);
            ++*(_DWORD *)(a1 + 44);
          }
          else
          {
            while ( v33 != -8 )
            {
              v31 = v30 & (v29 + v31);
              v32 = (__int64 *)(v28 + 16LL * v31);
              v33 = *v32;
              if ( *v87 == *v32 )
                goto LABEL_24;
              ++v29;
            }
          }
        }
        if ( i == v59 - (unsigned __int64)(v59 - 1) )
          break;
      }
    }
    if ( a3 != v65 )
    {
      v36 = &v76;
      v60 = *(_DWORD *)(v65 + 16);
      v55 = *(__int64 **)(v65 + 8);
      sub_15CEB90((__int64)&v73);
      v67 = *(char **)v65;
      v89[0] = v67;
      v69 = v67;
      v87 = v89;
      v88 = 0x4000000001LL;
      if ( (unsigned __int8)sub_15CE630((__int64)&v76, (__int64 *)&v69, &v81) )
        *((_DWORD *)sub_15D1D60((__int64)&v76, (__int64 *)&v67) + 3) = 0;
      v37 = v88;
      v66 = 0;
      if ( !(_DWORD)v88 )
      {
LABEL_52:
        if ( v87 != v89 )
          _libc_free((unsigned __int64)v87);
        sub_15D2F60((__int64 *)&v73, a1, v60);
        sub_15D2410(&v73, a1, v55);
        if ( v79 )
        {
          v47 = v77;
          v48 = &v77[9 * v79];
          do
          {
            if ( *v47 != -16 && *v47 != -8 )
            {
              v49 = v47[5];
              if ( (_QWORD *)v49 != v47 + 7 )
                _libc_free(v49);
            }
            v47 += 9;
          }
          while ( v48 != v47 );
        }
        goto LABEL_61;
      }
LABEL_36:
      while ( 2 )
      {
        v38 = (char *)v87[v37 - 1];
        LODWORD(v88) = v37 - 1;
        v68 = v38;
        v39 = sub_15D1D60((__int64)v36, (__int64 *)&v68);
        if ( *((_DWORD *)v39 + 2) )
        {
LABEL_35:
          v37 = v88;
          if ( !(_DWORD)v88 )
            goto LABEL_52;
          continue;
        }
        break;
      }
      *((_DWORD *)v39 + 4) = ++v66;
      *((_DWORD *)v39 + 2) = v66;
      v39[3] = (__int64)v68;
      sub_15CE600((__int64)&v73, &v68);
      sub_15CF6C0((__int64)&v81, (__int64)v68, v80);
      v40 = &v81[v82];
      if ( v81 == v40 )
        goto LABEL_50;
      v58 = (__int64)v36;
      v41 = v81;
      while ( 1 )
      {
        v43 = *v41;
        v69 = *v41;
        if ( !v79 )
          goto LABEL_39;
        v44 = (v79 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v45 = (char **)&v77[9 * v44];
        v46 = *v45;
        if ( v43 != *v45 )
          break;
LABEL_45:
        if ( v45 == &v77[9 * v79] || !*((_DWORD *)v45 + 2) )
          goto LABEL_39;
        if ( v43 == v68 )
        {
LABEL_42:
          if ( v40 == ++v41 )
            goto LABEL_49;
        }
        else
        {
          ++v41;
          sub_15CDD90((__int64)(v45 + 5), &v68);
          if ( v40 == v41 )
          {
LABEL_49:
            v36 = (__int64 *)v58;
            v40 = v81;
LABEL_50:
            if ( v40 != (char **)&v83 )
            {
              _libc_free((unsigned __int64)v40);
              v37 = v88;
              if ( !(_DWORD)v88 )
                goto LABEL_52;
              goto LABEL_36;
            }
            goto LABEL_35;
          }
        }
      }
      v50 = 1;
      while ( v46 != (char *)-8LL )
      {
        v51 = v50 + 1;
        v44 = (v79 - 1) & (v50 + v44);
        v45 = (char **)&v77[9 * v44];
        v46 = *v45;
        if ( v43 == *v45 )
          goto LABEL_45;
        v50 = v51;
      }
LABEL_39:
      v42 = sub_15CC510(a1, (__int64)v43);
      if ( v42 && v60 < *(_DWORD *)(v42 + 16) )
      {
        v63 = sub_15D1D60(v58, (__int64 *)&v69);
        sub_15CDD90((__int64)&v87, &v69);
        *((_DWORD *)v63 + 3) = v66;
        sub_15CDD90((__int64)(v63 + 5), &v68);
      }
      goto LABEL_42;
    }
  }
  else
  {
    sub_15D3360(a1, a2);
  }
  if ( v79 )
  {
    v52 = v77;
    v53 = &v77[9 * v79];
    do
    {
      if ( *v52 != -16 && *v52 != -8 )
      {
        v54 = v52[5];
        if ( (_QWORD *)v54 != v52 + 7 )
          _libc_free(v54);
      }
      v52 += 9;
    }
    while ( v53 != v52 );
  }
LABEL_61:
  j___libc_free_0(v77);
  sub_15CE080(&v73);
  if ( v84 != (__int64 *)v86 )
    _libc_free((unsigned __int64)v84);
}
