// Function: sub_39D4670
// Address: 0x39d4670
//
__int64 __fastcall sub_39D4670(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r9d
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // r14
  char *v11; // rax
  size_t v12; // rdx
  void *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rdi
  const char *v16; // rsi
  __int64 v17; // rdi
  void *v18; // rdx
  __int64 v19; // rdi
  const char *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // r12
  const char *v25; // rsi
  __int64 v26; // rdi
  void *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdx
  _BYTE *v30; // rax
  _WORD *v31; // rdx
  __int64 v32; // rax
  void *v33; // rdx
  __int64 *v34; // r12
  __int64 v35; // rsi
  char **v36; // rdi
  __int64 v37; // r15
  __int64 v38; // rdx
  __int64 v39; // r8
  _BYTE *v40; // rax
  int v41; // r9d
  __int64 v42; // rdi
  _BYTE *v43; // rax
  __int64 v44; // rdi
  _WORD *v45; // rdx
  __int64 v46; // rdi
  _BYTE *v47; // rax
  __int64 result; // rax
  __int64 (*v49)(void); // rax
  __int64 v50; // rax
  unsigned __int16 *v51; // r14
  unsigned __int16 *v52; // r12
  __int64 v53; // r14
  __int64 v54; // rcx
  int v55; // r9d
  __int64 v56; // r8
  __int64 v57; // rdx
  __int64 v58; // rdi
  _WORD *v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // r12
  __int64 v62; // r13
  char v63; // r14
  __int64 v64; // rdi
  __int64 v65; // rdi
  __int64 v66; // rdi
  _WORD *v67; // rdx
  __int64 v68; // rax
  _WORD *v69; // rdx
  __int64 v70; // rax
  char v71; // [rsp+Fh] [rbp-A1h]
  __int64 *v72; // [rsp+10h] [rbp-A0h]
  unsigned __int16 *v73; // [rsp+10h] [rbp-A0h]
  __int64 v74; // [rsp+18h] [rbp-98h]
  __int64 v75; // [rsp+18h] [rbp-98h]
  size_t v76; // [rsp+18h] [rbp-98h]
  __int64 v77; // [rsp+20h] [rbp-90h]
  _QWORD v78[2]; // [rsp+40h] [rbp-70h] BYREF
  void (__fastcall *v79)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-60h]
  void (__fastcall *v80)(_QWORD *, __int64); // [rsp+58h] [rbp-58h]
  char *v81; // [rsp+60h] [rbp-50h] BYREF
  const char *v82; // [rsp+68h] [rbp-48h]
  __int64 (__fastcall *v83)(char **, char **, int); // [rsp+70h] [rbp-40h] BYREF
  __int64 (__fastcall *v84)(int *, __int64, __int64, __int64, __int64, int); // [rsp+78h] [rbp-38h]

  v4 = *a1;
  v5 = *(_QWORD *)(v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v5) <= 2 )
  {
    v4 = sub_16E7EE0(v4, "bb.", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 46;
    *(_WORD *)v5 = 25186;
    *(_QWORD *)(v4 + 24) += 3LL;
  }
  sub_16E7AB0(v4, *(int *)(a2 + 48));
  v8 = *(_QWORD *)(a2 + 40);
  if ( !v8 )
    goto LABEL_8;
  v9 = *a1;
  if ( (*(_BYTE *)(v8 + 23) & 0x20) != 0 )
  {
    v10 = sub_1263B40(v9, ".");
    v11 = (char *)sub_1649960(v8);
    v13 = *(void **)(v10 + 24);
    if ( v12 > *(_QWORD *)(v10 + 16) - (_QWORD)v13 )
    {
      sub_16E7EE0(v10, v11, v12);
      v14 = *a1;
      v15 = *a1;
      if ( *(_BYTE *)(a2 + 181) )
        goto LABEL_9;
      goto LABEL_22;
    }
    if ( v12 )
    {
      v76 = v12;
      memcpy(v13, v11, v12);
      *(_QWORD *)(v10 + 24) += v76;
    }
LABEL_8:
    v14 = *a1;
    v15 = *a1;
    if ( *(_BYTE *)(a2 + 181) )
    {
LABEL_9:
      v16 = " (";
      goto LABEL_10;
    }
LABEL_22:
    v25 = " (";
    if ( !*(_BYTE *)(a2 + 180) )
    {
      if ( !*(_DWORD *)(a2 + 176) )
        goto LABEL_37;
      v20 = " (";
      goto LABEL_30;
    }
    goto LABEL_26;
  }
  sub_1263B40(v9, " (");
  v23 = sub_154F480(a1[1], v8, v21, v22);
  if ( v23 == -1 )
  {
    sub_1263B40(*a1, "<ir-block badref>");
    if ( !*(_BYTE *)(a2 + 181) )
    {
      if ( *(_BYTE *)(a2 + 180) )
      {
        v14 = *a1;
        v25 = ", ";
        goto LABEL_26;
      }
      v19 = *a1;
      goto LABEL_20;
    }
    v14 = *a1;
    v16 = ", ";
  }
  else
  {
    LODWORD(v77) = v23;
    v24 = *a1;
    v78[0] = "%ir-block.";
    v78[1] = v77;
    LOWORD(v79) = 2563;
    sub_16E2FC0((__int64 *)&v81, (__int64)v78);
    sub_16E7EE0(v24, v81, (size_t)v82);
    if ( v81 != (char *)&v83 )
      j_j___libc_free_0((unsigned __int64)v81);
    v14 = *a1;
    v19 = *a1;
    if ( !*(_BYTE *)(a2 + 181) )
    {
      if ( *(_BYTE *)(a2 + 180) )
      {
LABEL_25:
        v25 = ", ";
LABEL_26:
        sub_1263B40(v14, v25);
        v26 = *a1;
        v27 = *(void **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v27 <= 0xAu )
        {
          sub_16E7EE0(v26, "landing-pad", 0xBu);
        }
        else
        {
          qmemcpy(v27, "landing-pad", 11);
          *(_QWORD *)(v26 + 24) += 11LL;
        }
        if ( !*(_DWORD *)(a2 + 176) )
          goto LABEL_33;
        v14 = *a1;
        v20 = ", ";
LABEL_30:
        sub_1263B40(v14, v20);
        v28 = *a1;
        v29 = *(_QWORD *)(*a1 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v29) <= 5 )
        {
          v28 = sub_16E7EE0(v28, "align ", 6u);
        }
        else
        {
          *(_DWORD *)v29 = 1734962273;
          *(_WORD *)(v29 + 4) = 8302;
          *(_QWORD *)(v28 + 24) += 6LL;
        }
        sub_16E7A90(v28, *(unsigned int *)(a2 + 176));
LABEL_33:
        v19 = *a1;
        goto LABEL_34;
      }
LABEL_20:
      v14 = v19;
      v20 = ", ";
      if ( *(_DWORD *)(a2 + 176) )
        goto LABEL_30;
      goto LABEL_34;
    }
    v16 = ", ";
  }
LABEL_10:
  sub_1263B40(v14, v16);
  v17 = *a1;
  v18 = *(void **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v18 <= 0xCu )
  {
    sub_16E7EE0(v17, "address-taken", 0xDu);
  }
  else
  {
    qmemcpy(v18, "address-taken", 13);
    *(_QWORD *)(v17 + 24) += 13LL;
  }
  v14 = *a1;
  v19 = *a1;
  if ( *(_BYTE *)(a2 + 180) )
    goto LABEL_25;
  if ( *(_DWORD *)(a2 + 176) )
  {
    v20 = ", ";
    goto LABEL_30;
  }
LABEL_34:
  v30 = *(_BYTE **)(v19 + 24);
  if ( *(_BYTE **)(v19 + 16) == v30 )
  {
    sub_16E7EE0(v19, ")", 1u);
  }
  else
  {
    *v30 = 41;
    ++*(_QWORD *)(v19 + 24);
  }
  v15 = *a1;
LABEL_37:
  v31 = *(_WORD **)(v15 + 24);
  if ( *(_QWORD *)(v15 + 16) - (_QWORD)v31 <= 1u )
  {
    sub_16E7EE0(v15, ":\n", 2u);
  }
  else
  {
    *v31 = 2618;
    *(_QWORD *)(v15 + 24) += 2LL;
  }
  v71 = sub_39D4490((__int64)a1, (_QWORD *)a2, (__int64)v31, v6, v14, v7);
  if ( *(_QWORD *)(a2 + 88) != *(_QWORD *)(a2 + 96) && !byte_50576C0
    || !v71
    || !(unsigned __int8)sub_39D3680((__int64)a1, (_QWORD *)a2) )
  {
    v32 = sub_16E8750(*a1, 2u);
    v33 = *(void **)(v32 + 24);
    if ( *(_QWORD *)(v32 + 16) - (_QWORD)v33 <= 0xBu )
    {
      sub_16E7EE0(v32, "successors: ", 0xCu);
    }
    else
    {
      qmemcpy(v33, "successors: ", 12);
      *(_QWORD *)(v32 + 24) += 12LL;
    }
    v34 = *(__int64 **)(a2 + 88);
    v72 = *(__int64 **)(a2 + 96);
    if ( v34 == v72 )
    {
LABEL_59:
      v46 = *a1;
      v47 = *(_BYTE **)(*a1 + 24);
      if ( *(_BYTE **)(*a1 + 16) == v47 )
      {
        sub_16E7EE0(v46, "\n", 1u);
      }
      else
      {
        *v47 = 10;
        ++*(_QWORD *)(v46 + 24);
      }
      result = **(_QWORD **)(*(_QWORD *)(a2 + 56) + 40LL);
      if ( (**(_BYTE **)(result + 352) & 4) == 0 || *(_QWORD *)(a2 + 160) == *(_QWORD *)(a2 + 152) )
      {
LABEL_80:
        v60 = *a1;
        result = *(_QWORD *)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) == result )
        {
          result = sub_16E7EE0(v60, "\n", 1u);
        }
        else
        {
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(v60 + 24);
        }
        goto LABEL_82;
      }
LABEL_63:
      v75 = 0;
      v49 = *(__int64 (**)(void))(**(_QWORD **)(result + 16) + 112LL);
      if ( v49 != sub_1D00B10 )
        v75 = v49();
      v50 = sub_16E8750(*a1, 2u);
      sub_1263B40(v50, "liveins: ");
      v51 = *(unsigned __int16 **)(a2 + 160);
      v73 = v51;
      v52 = (unsigned __int16 *)sub_1DD77D0(a2);
      if ( v51 != v52 )
      {
        while ( 1 )
        {
          v35 = *v52;
          v36 = &v81;
          v53 = *a1;
          sub_1F4AA00((__int64 *)&v81, v35, v75, 0, 0);
          if ( !v83 )
            break;
          ((void (__fastcall *)(char **, __int64))v84)(&v81, v53);
          if ( v83 )
            v83(&v81, &v81, 3);
          if ( *((_DWORD *)v52 + 1) != -1 )
          {
            v56 = *a1;
            v57 = *(_QWORD *)(*a1 + 24);
            if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v57) <= 2 )
            {
              v56 = sub_16E7EE0(*a1, ":0x", 3u);
            }
            else
            {
              *(_BYTE *)(v57 + 2) = 120;
              *(_WORD *)v57 = 12346;
              *(_QWORD *)(v56 + 24) += 3LL;
            }
            LODWORD(v81) = *((_DWORD *)v52 + 1);
            v83 = (__int64 (__fastcall *)(char **, char **, int))sub_1DB3470;
            v84 = sub_1DB3430;
            sub_1DB3430((int *)&v81, v56, v57, v54, v56, v55);
            if ( v83 )
              v83(&v81, &v81, 3);
          }
          v52 += 4;
          if ( v52 == v73 )
            goto LABEL_79;
          v58 = *a1;
          v59 = *(_WORD **)(*a1 + 24);
          if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v59 <= 1u )
          {
            sub_16E7EE0(v58, ", ", 2u);
          }
          else
          {
            *v59 = 8236;
            *(_QWORD *)(v58 + 24) += 2LL;
          }
        }
LABEL_125:
        sub_4263D6(v36, v35, v38);
      }
LABEL_79:
      sub_1263B40(*a1, "\n");
      goto LABEL_80;
    }
    while ( 1 )
    {
      v35 = *v34;
      v36 = (char **)v78;
      v37 = *a1;
      sub_1DD5B60(v78, *v34);
      if ( !v79 )
        goto LABEL_125;
      v80(v78, v37);
      if ( v79 )
        v79(v78, v78, 3);
      if ( byte_50576C0 && v71 )
        goto LABEL_54;
      v39 = *a1;
      v40 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v40 >= *(_QWORD *)(*a1 + 16) )
      {
        v39 = sub_16E7DE0(*a1, 40);
      }
      else
      {
        *(_QWORD *)(v39 + 24) = v40 + 1;
        *v40 = 40;
      }
      v74 = v39;
      LODWORD(v83) = sub_1DD75B0((_QWORD *)a2, (__int64)v34);
      v82 = "0x%08x";
      v81 = (char *)&unk_49EFAC8;
      v42 = sub_16E8450(v74, (__int64)&v81, (__int64)&unk_49EFAC8, (__int64)"0x%08x", v74, v41);
      v43 = *(_BYTE **)(v42 + 24);
      if ( (unsigned __int64)v43 < *(_QWORD *)(v42 + 16) )
        break;
      ++v34;
      sub_16E7DE0(v42, 41);
      if ( v72 == v34 )
        goto LABEL_59;
LABEL_55:
      if ( *(__int64 **)(a2 + 88) != v34 )
      {
        v44 = *a1;
        v45 = *(_WORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v45 <= 1u )
        {
          sub_16E7EE0(v44, ", ", 2u);
        }
        else
        {
          *v45 = 8236;
          *(_QWORD *)(v44 + 24) += 2LL;
        }
      }
    }
    *(_QWORD *)(v42 + 24) = v43 + 1;
    *v43 = 41;
LABEL_54:
    if ( v72 == ++v34 )
      goto LABEL_59;
    goto LABEL_55;
  }
  result = **(_QWORD **)(*(_QWORD *)(a2 + 56) + 40LL);
  if ( (**(_BYTE **)(result + 352) & 4) != 0 && *(_QWORD *)(a2 + 160) != *(_QWORD *)(a2 + 152) )
    goto LABEL_63;
LABEL_82:
  v61 = *(_QWORD *)(a2 + 32);
  v62 = a2 + 24;
  v63 = 0;
  if ( v62 == v61 )
    return result;
  do
  {
    while ( 1 )
    {
      v65 = *a1;
      if ( v63 )
      {
        if ( (*(_BYTE *)(v61 + 46) & 4) != 0 )
        {
          sub_16E8750(v65, 4u);
          sub_39D3D30(a1, v61);
          goto LABEL_86;
        }
        v68 = sub_16E8750(v65, 2u);
        v69 = *(_WORD **)(v68 + 24);
        if ( *(_QWORD *)(v68 + 16) - (_QWORD)v69 <= 1u )
        {
          sub_16E7EE0(v68, "}\n", 2u);
        }
        else
        {
          *v69 = 2685;
          *(_QWORD *)(v68 + 24) += 2LL;
        }
        v65 = *a1;
      }
      v63 = 0;
      sub_16E8750(v65, 2u);
      sub_39D3D30(a1, v61);
      if ( (*(_BYTE *)(v61 + 46) & 8) != 0 )
        break;
LABEL_86:
      v64 = *a1;
      result = *(_QWORD *)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) == result )
        goto LABEL_93;
LABEL_87:
      *(_BYTE *)result = 10;
      ++*(_QWORD *)(v64 + 24);
      v61 = *(_QWORD *)(v61 + 8);
      if ( v62 == v61 )
        goto LABEL_94;
    }
    v66 = *a1;
    v67 = *(_WORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v67 <= 1u )
    {
      sub_16E7EE0(v66, " {", 2u);
    }
    else
    {
      *v67 = 31520;
      *(_QWORD *)(v66 + 24) += 2LL;
    }
    v64 = *a1;
    v63 = 1;
    result = *(_QWORD *)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) != result )
      goto LABEL_87;
LABEL_93:
    result = sub_16E7EE0(v64, "\n", 1u);
    v61 = *(_QWORD *)(v61 + 8);
  }
  while ( v62 != v61 );
LABEL_94:
  if ( v63 )
  {
    v70 = sub_16E8750(*a1, 2u);
    return sub_1263B40(v70, "}\n");
  }
  return result;
}
