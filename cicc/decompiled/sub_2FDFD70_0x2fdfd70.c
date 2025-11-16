// Function: sub_2FDFD70
// Address: 0x2fdfd70
//
__int64 __fastcall sub_2FDFD70(__int64 *a1, __int64 a2, char *a3, unsigned __int64 a4, __int64 a5)
{
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rsi
  _QWORD *v11; // r15
  __int64 (*v12)(); // rax
  __int64 (*v13)(); // rax
  __int64 (*v14)(); // rax
  __int64 v15; // r13
  __int64 v17; // rax
  int *v18; // rcx
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // r8
  void *v24; // r9
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 *v27; // rbx
  int *v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rcx
  int v34; // eax
  int v35; // eax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  int *v41; // rdx
  __int64 v42; // [rsp+8h] [rbp-58h]
  unsigned __int64 v43; // [rsp+8h] [rbp-58h]
  char *v44; // [rsp+10h] [rbp-50h]
  unsigned int *v45; // [rsp+10h] [rbp-50h]
  int v46[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = *(unsigned __int16 *)(a2 + 68);
  v9 = *(_QWORD *)(a2 + 24);
  v46[0] = 0;
  v10 = *a1;
  v11 = *(_QWORD **)(v9 + 32);
  if ( (((_WORD)v8 - 26) & 0xFFFD) == 0 || (_WORD)v8 == 32 )
  {
    v12 = *(__int64 (**)())(v10 + 88);
    if ( v12 != sub_2E97330 )
    {
      v42 = a4;
      v44 = a3;
      v35 = ((__int64 (__fastcall *)(__int64 *, __int64, int *))v12)(a1, a5, v46);
      a3 = v44;
      a4 = v42;
      if ( v35 )
      {
        v36 = sub_2FDE160(v11, a2, v44, v42, v46[0], a1);
        v15 = (__int64)v36;
        if ( v36 )
        {
          sub_2E31040((__int64 *)(v9 + 40), (__int64)v36);
          v37 = *(_QWORD *)a2;
          v38 = *(_QWORD *)v15;
          *(_QWORD *)(v15 + 8) = a2;
          v37 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v15 = v37 | v38 & 7;
          *(_QWORD *)(v37 + 8) = v15;
          *(_QWORD *)a2 = v15 | *(_QWORD *)a2 & 7LL;
          goto LABEL_10;
        }
        return 0;
      }
      v10 = *a1;
      v8 = *(unsigned __int16 *)(a2 + 68);
    }
  }
  if ( (unsigned int)(v8 - 1) <= 1 )
  {
    v13 = *(__int64 (**)())(v10 + 88);
    if ( v13 != sub_2E97330 )
    {
      v43 = a4;
      v45 = (unsigned int *)a3;
      v39 = ((__int64 (__fastcall *)(__int64 *, __int64, int *))v13)(a1, a5, v46);
      a3 = (char *)v45;
      a4 = v43;
      if ( v39 )
        return sub_2FDD0C0((__int64 *)a2, v45, v43, v46[0], a1);
      v10 = *a1;
    }
  }
  v14 = *(__int64 (**)())(v10 + 744);
  if ( v14 == sub_2FDC640 )
    return 0;
  v15 = ((__int64 (__fastcall *)(__int64 *, _QWORD *, __int64, char *, unsigned __int64, __int64, __int64))v14)(
          a1,
          v11,
          a2,
          a3,
          a4,
          a2,
          a5);
  if ( !v15 )
    return 0;
LABEL_10:
  v17 = *(_QWORD *)(a2 + 48);
  v18 = (int *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_23;
  if ( (v17 & 7) != 0 )
  {
    if ( (v17 & 7) == 3 && *v18 )
      goto LABEL_14;
LABEL_23:
    v31 = *(_QWORD *)(a5 + 48);
    v32 = 0;
    v33 = v31 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v31 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v34 = v31 & 7;
      if ( v34 )
      {
        if ( v34 == 3 )
        {
          v33 = *(int *)v33;
          v32 = (_QWORD *)((*(_QWORD *)(a5 + 48) & 0xFFFFFFFFFFFFFFF8LL) + 16);
        }
        else
        {
          v33 = 0;
        }
      }
      else
      {
        *(_QWORD *)(a5 + 48) = v33;
        v32 = (_QWORD *)(a5 + 48);
        v33 = 1;
      }
    }
    sub_2E86A90(v15, (__int64)v11, v32, v33);
    return v15;
  }
  *(_QWORD *)(a2 + 48) = v18;
  LOBYTE(v17) = v17 & 0xF8;
LABEL_14:
  v19 = v17 & 7;
  if ( v19 )
  {
    if ( v19 == 3 )
    {
      v41 = v18;
      v21 = *v18;
      v20 = v41 + 4;
    }
    else
    {
      v21 = 0;
      v20 = 0;
    }
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v18;
    v20 = (_QWORD *)(a2 + 48);
    v21 = 1;
  }
  sub_2E86A90(v15, (__int64)v11, v20, v21);
  v25 = *(_QWORD *)(a5 + 48);
  v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v22 = v25 & 7;
    if ( (v25 & 7) == 0 )
    {
      *(_QWORD *)(a5 + 48) = v26;
      v27 = (__int64 *)(a5 + 48);
      v25 &= 0xFFFFFFFFFFFFFFF8LL;
      goto LABEL_19;
    }
    v27 = (__int64 *)(v26 + 16);
    if ( (_DWORD)v22 == 3 )
    {
LABEL_19:
      v28 = (int *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        v29 = 0;
        do
        {
LABEL_21:
          v30 = *v27++;
          sub_2E86C70(v15, (__int64)v11, v30, v22, v23, v24);
        }
        while ( v27 != (__int64 *)v29 );
        return v15;
      }
      goto LABEL_36;
    }
  }
  v28 = (int *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return v15;
  v27 = 0;
LABEL_36:
  v40 = v25 & 7;
  if ( v40 )
  {
    v29 = 0;
    if ( v40 == 3 )
      v29 = (__int64)&v28[2 * *v28 + 4];
  }
  else
  {
    *(_QWORD *)(a5 + 48) = v28;
    v29 = a5 + 56;
  }
  if ( (__int64 *)v29 != v27 )
    goto LABEL_21;
  return v15;
}
