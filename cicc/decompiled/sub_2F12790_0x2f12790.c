// Function: sub_2F12790
// Address: 0x2f12790
//
_BYTE *__fastcall sub_2F12790(__int64 *a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdi
  _WORD *v8; // rdx
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  void *v12; // rdx
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int *v17; // r12
  __int64 v18; // rsi
  __int64 *v19; // rdi
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rdi
  _WORD *v23; // rdx
  _BYTE *result; // rax
  __int64 v25; // r14
  __int64 v26; // r12
  char v27; // r13
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  _WORD *v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rax
  _WORD *v36; // rdx
  __int64 *v37; // r12
  __int64 v38; // r15
  __int64 v39; // r8
  _BYTE *v40; // rax
  __int64 v41; // r9
  __int64 v42; // rdi
  _BYTE *v43; // rax
  __int64 v44; // rdi
  _WORD *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // [rsp+8h] [rbp-98h]
  __int64 v48; // [rsp+10h] [rbp-90h]
  char v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+18h] [rbp-88h]
  unsigned int *v51; // [rsp+20h] [rbp-80h]
  __int64 *v52; // [rsp+20h] [rbp-80h]
  __int64 v53; // [rsp+28h] [rbp-78h]
  __int64 v54; // [rsp+28h] [rbp-78h]
  _QWORD v55[2]; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v56)(_QWORD *, _QWORD *, __int64); // [rsp+40h] [rbp-60h]
  void (__fastcall *v57)(_QWORD *, __int64); // [rsp+48h] [rbp-58h]
  __int64 v58[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 (__fastcall *v59)(unsigned __int64 *, const __m128i **, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v60)(__int64 *, __int64); // [rsp+68h] [rbp-38h]

  sub_2E37380(a2, *a1, 3, a1[1]);
  v7 = *a1;
  v8 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v7, (unsigned __int8 *)":\n", 2u);
  }
  else
  {
    *v8 = 2618;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v49 = sub_2F125C0((__int64)a1, a2, (__int64)v8, v4, v5, v6);
  if ( (!*(_DWORD *)(a2 + 120) || (_BYTE)qword_50225C8) && v49 && (unsigned __int8)sub_2F10E20((__int64)a1, a2, v9, v10) )
  {
    result = *(_BYTE **)(a2 + 192);
    if ( *(_BYTE **)(a2 + 184) == result )
    {
      v25 = a2 + 48;
      goto LABEL_24;
    }
    goto LABEL_11;
  }
  v11 = sub_CB69B0(*a1, 2u);
  v12 = *(void **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xAu )
  {
    sub_CB6200(v11, "successors:", 0xBu);
  }
  else
  {
    qmemcpy(v12, "successors:", 11);
    *(_QWORD *)(v11 + 32) += 11LL;
  }
  v13 = *a1;
  if ( *(_DWORD *)(a2 + 120) )
  {
    sub_904010(v13, " ");
    v37 = *(__int64 **)(a2 + 112);
    v52 = &v37[*(unsigned int *)(a2 + 120)];
    if ( v37 == v52 )
    {
LABEL_65:
      v13 = *a1;
      v14 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) != v14 )
        goto LABEL_9;
      goto LABEL_66;
    }
    while ( 1 )
    {
      v18 = *v37;
      v19 = v55;
      v38 = *a1;
      sub_2E31000(v55, *v37);
      if ( !v56 )
        goto LABEL_76;
      v57(v55, v38);
      if ( v56 )
        v56(v55, v55, 3);
      if ( (_BYTE)qword_50225C8 && v49 )
        goto LABEL_60;
      v39 = *a1;
      v40 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v40 >= *(_QWORD *)(*a1 + 24) )
      {
        v39 = sub_CB5D20(*a1, 40);
      }
      else
      {
        *(_QWORD *)(v39 + 32) = v40 + 1;
        *v40 = 40;
      }
      v54 = v39;
      LODWORD(v59) = sub_2E32EA0(a2, (__int64)v37);
      v58[1] = (__int64)"0x%08x";
      v58[0] = (__int64)&unk_49DD0F8;
      v42 = sub_CB6620(v54, (__int64)v58, (__int64)&unk_49DD0F8, (__int64)"0x%08x", v54, v41);
      v43 = *(_BYTE **)(v42 + 32);
      if ( (unsigned __int64)v43 < *(_QWORD *)(v42 + 24) )
        break;
      ++v37;
      sub_CB5D20(v42, 41);
      if ( v37 == v52 )
        goto LABEL_65;
LABEL_61:
      if ( *(__int64 **)(a2 + 112) != v37 )
      {
        v44 = *a1;
        v45 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v45 <= 1u )
        {
          sub_CB6200(v44, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v45 = 8236;
          *(_QWORD *)(v44 + 32) += 2LL;
        }
      }
    }
    *(_QWORD *)(v42 + 32) = v43 + 1;
    *v43 = 41;
LABEL_60:
    if ( ++v37 == v52 )
      goto LABEL_65;
    goto LABEL_61;
  }
  v14 = *(_BYTE **)(v13 + 32);
  if ( *(_BYTE **)(v13 + 24) != v14 )
  {
LABEL_9:
    *v14 = 10;
    ++*(_QWORD *)(v13 + 32);
    goto LABEL_10;
  }
LABEL_66:
  sub_CB6200(v13, (unsigned __int8 *)"\n", 1u);
LABEL_10:
  if ( *(_QWORD *)(a2 + 192) != *(_QWORD *)(a2 + 184) )
  {
LABEL_11:
    v15 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a2 + 32) + 32LL) + 16LL);
    v53 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 200LL))(v15);
    v16 = sub_CB69B0(*a1, 2u);
    sub_904010(v16, "liveins: ");
    v17 = *(unsigned int **)(a2 + 184);
    v51 = *(unsigned int **)(a2 + 192);
    if ( v51 != v17 )
    {
      while ( 1 )
      {
        v18 = *v17;
        v19 = v58;
        v20 = *a1;
        sub_2FF6320(v58, v18, v53, 0, 0);
        if ( !v59 )
          break;
        v60(v58, v20);
        if ( v59 )
          v59((unsigned __int64 *)v58, (const __m128i **)v58, 3);
        if ( *((_QWORD *)v17 + 1) == -1 && *((_QWORD *)v17 + 2) == -1 )
          goto LABEL_17;
        v32 = *a1;
        v33 = *(_QWORD *)(*a1 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v33) <= 2 )
        {
          v32 = sub_CB6200(*a1, ":0x", 3u);
        }
        else
        {
          *(_BYTE *)(v33 + 2) = 120;
          *(_WORD *)v33 = 12346;
          *(_QWORD *)(v32 + 32) += 3LL;
        }
        v47 = v32;
        v48 = *((_QWORD *)v17 + 1);
        v50 = *((_QWORD *)v17 + 2);
        v34 = (_QWORD *)sub_22077B0(0x10u);
        if ( v34 )
        {
          *v34 = v48;
          v34[1] = v50;
        }
        v58[0] = (__int64)v34;
        v59 = sub_2E09350;
        v60 = sub_2E092F0;
        sub_2E092F0(v58, v47);
        if ( !v59 )
        {
LABEL_17:
          v17 += 6;
          if ( v51 == v17 )
            goto LABEL_44;
        }
        else
        {
          v17 += 6;
          v59((unsigned __int64 *)v58, (const __m128i **)v58, 3);
          if ( v51 == v17 )
            goto LABEL_44;
        }
        v22 = *a1;
        v23 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v23 <= 1u )
        {
          sub_CB6200(v22, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v23 = 8236;
          *(_QWORD *)(v22 + 32) += 2LL;
        }
      }
LABEL_76:
      sub_4263D6(v19, v18, v21);
    }
LABEL_44:
    sub_904010(*a1, "\n");
  }
  v25 = a2 + 48;
  result = (_BYTE *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_BYTE *)(a2 + 48) != result )
    result = (_BYTE *)sub_904010(*a1, "\n");
LABEL_24:
  v26 = *(_QWORD *)(a2 + 56);
  v27 = 0;
  if ( v25 == v26 )
    return result;
  do
  {
    while ( 1 )
    {
      v29 = *a1;
      if ( v27 )
      {
        if ( (*(_BYTE *)(v26 + 44) & 4) != 0 )
        {
          sub_CB69B0(v29, 4u);
          sub_2F116B0(a1, v26);
          goto LABEL_28;
        }
        v35 = sub_CB69B0(v29, 2u);
        v36 = *(_WORD **)(v35 + 32);
        if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 1u )
        {
          sub_CB6200(v35, "}\n", 2u);
        }
        else
        {
          *v36 = 2685;
          *(_QWORD *)(v35 + 32) += 2LL;
        }
        v29 = *a1;
      }
      v27 = 0;
      sub_CB69B0(v29, 2u);
      sub_2F116B0(a1, v26);
      if ( (*(_BYTE *)(v26 + 44) & 8) != 0 )
        break;
LABEL_28:
      v28 = *a1;
      result = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) == result )
        goto LABEL_35;
LABEL_29:
      *result = 10;
      ++*(_QWORD *)(v28 + 32);
      v26 = *(_QWORD *)(v26 + 8);
      if ( v25 == v26 )
        goto LABEL_36;
    }
    v30 = *a1;
    v31 = *(_WORD **)(*a1 + 32);
    v27 = 1;
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v31 <= 1u )
    {
      sub_CB6200(v30, (unsigned __int8 *)" {", 2u);
    }
    else
    {
      *v31 = 31520;
      *(_QWORD *)(v30 + 32) += 2LL;
    }
    v28 = *a1;
    result = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) != result )
      goto LABEL_29;
LABEL_35:
    result = (_BYTE *)sub_CB6200(v28, (unsigned __int8 *)"\n", 1u);
    v26 = *(_QWORD *)(v26 + 8);
  }
  while ( v25 != v26 );
LABEL_36:
  if ( v27 )
  {
    v46 = sub_CB69B0(*a1, 2u);
    return (_BYTE *)sub_904010(v46, "}\n");
  }
  return result;
}
