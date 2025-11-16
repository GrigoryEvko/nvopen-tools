// Function: sub_2B5AE70
// Address: 0x2b5ae70
//
void __fastcall sub_2B5AE70(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v20; // r9
  unsigned int v21; // esi
  __int64 v22; // r15
  __int64 v23; // r8
  __int64 v24; // rdi
  int v25; // r11d
  __int64 v26; // rcx
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r10
  unsigned __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  int v33; // eax
  int v34; // edx
  __int64 v35; // rdi
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rsi
  __int64 v39; // rsi
  _QWORD *v40; // rdx
  _BYTE *v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-120h]
  __int64 v44; // [rsp+10h] [rbp-110h]
  __int64 v45; // [rsp+10h] [rbp-110h]
  __int64 v46; // [rsp+10h] [rbp-110h]
  __int64 v47; // [rsp+10h] [rbp-110h]
  __int64 v48; // [rsp+18h] [rbp-108h]
  __int64 v49; // [rsp+28h] [rbp-F8h]
  unsigned __int8 *v50; // [rsp+30h] [rbp-F0h] BYREF
  int v51; // [rsp+38h] [rbp-E8h]
  char *v52; // [rsp+40h] [rbp-E0h]
  __int64 v53; // [rsp+48h] [rbp-D8h]
  char v54; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+90h] [rbp-90h] BYREF
  _BYTE *v56; // [rsp+98h] [rbp-88h]
  __int64 v57; // [rsp+A0h] [rbp-80h]
  _BYTE v58[120]; // [rsp+A8h] [rbp-78h] BYREF

  v49 = a1 + 72;
  sub_2B3FDA0(a1 + 72);
  v3 = *(_QWORD *)(a1 + 104);
  v48 = a1 + 104;
  v4 = v3 + 88LL * *(unsigned int *)(a1 + 112);
  while ( v3 != v4 )
  {
    while ( 1 )
    {
      v4 -= 88;
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 == v4 + 24 )
        break;
      _libc_free(v5);
      if ( v3 == v4 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(a1 + 112) = 0;
  sub_2B3FDA0(a1 + 120);
  v6 = *(_QWORD *)(a1 + 152);
  v7 = v6 + 88LL * *(unsigned int *)(a1 + 160);
  while ( v6 != v7 )
  {
    v7 -= 88;
    v8 = *(_QWORD *)(v7 + 8);
    if ( v8 != v7 + 24 )
      _libc_free(v8);
  }
  *(_DWORD *)(a1 + 160) = 0;
  v9 = *(_QWORD *)(a2 + 56);
  v10 = a2 + 48;
  if ( v9 != a2 + 48 )
  {
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      v12 = *(_BYTE *)(v9 - 24);
      v13 = v9 - 24;
      if ( v12 == 62 )
        break;
      if ( v12 == 63
        && (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) == 2
        && (v14 = *(_QWORD *)(v9 - 56), *(_BYTE *)v14 > 0x15u)
        && sub_2B08630(*(_QWORD *)(v14 + 8))
        && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 - 16) + 8LL) - 17 > 1 )
      {
        v55 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v9 - 20) & 0x7FFFFFF));
        v17 = sub_2B5ABC0(a1 + 120, &v55);
        v18 = *(unsigned int *)(v17 + 8);
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 12) )
        {
          sub_C8D5F0(v17, (const void *)(v17 + 16), v18 + 1, 8u, v15, v16);
          v18 = *(unsigned int *)(v17 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v17 + 8 * v18) = v13;
        ++*(_DWORD *)(v17 + 8);
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          return;
      }
      else
      {
LABEL_18:
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          return;
      }
    }
    if ( sub_B46500((unsigned __int8 *)(v9 - 24)) || (*(_BYTE *)(v9 - 22) & 1) != 0 )
      goto LABEL_18;
    v11 = *(_QWORD *)(*(_QWORD *)(v9 - 88) + 8LL);
    if ( (_BYTE)qword_5010508 && *(_BYTE *)(v11 + 8) == 17 )
      v11 = **(_QWORD **)(v11 + 16);
    if ( !(unsigned __int8)sub_BCBCB0(v11) || (*(_BYTE *)(v11 + 8) & 0xFD) == 4 )
      goto LABEL_18;
    v19 = sub_98ACB0(*(unsigned __int8 **)(v9 - 56), 6u);
    v21 = *(_DWORD *)(a1 + 96);
    v51 = 0;
    v50 = v19;
    v22 = (__int64)v19;
    if ( v21 )
    {
      v23 = v21 - 1;
      v24 = *(_QWORD *)(a1 + 80);
      v25 = 1;
      v26 = 0;
      v27 = v23 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v28 = v24 + 16LL * v27;
      v29 = *(_QWORD *)v28;
      if ( v22 == *(_QWORD *)v28 )
      {
LABEL_32:
        v30 = *(unsigned int *)(v28 + 8);
LABEL_33:
        v31 = *(_QWORD *)(a1 + 104) + 88 * v30;
        v32 = *(unsigned int *)(v31 + 16);
        if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 20) )
        {
          sub_C8D5F0(v31 + 8, (const void *)(v31 + 24), v32 + 1, 8u, v23, v20);
          v32 = *(unsigned int *)(v31 + 16);
        }
        *(_QWORD *)(*(_QWORD *)(v31 + 8) + 8 * v32) = v13;
        ++*(_DWORD *)(v31 + 16);
        goto LABEL_18;
      }
      while ( v29 != -4096 )
      {
        if ( !v26 && v29 == -8192 )
          v26 = v28;
        v20 = (unsigned int)(v25 + 1);
        v27 = v23 & (v25 + v27);
        v28 = v24 + 16LL * v27;
        v29 = *(_QWORD *)v28;
        if ( v22 == *(_QWORD *)v28 )
          goto LABEL_32;
        ++v25;
      }
      if ( !v26 )
        v26 = v28;
      v33 = *(_DWORD *)(a1 + 88);
      ++*(_QWORD *)(a1 + 72);
      v34 = v33 + 1;
      v55 = v26;
      if ( 4 * (v33 + 1) < 3 * v21 )
      {
        v35 = v22;
        v20 = v21 >> 3;
        v23 = (__int64)&v55;
        if ( v21 - *(_DWORD *)(a1 + 92) - v34 > (unsigned int)v20 )
        {
LABEL_46:
          *(_DWORD *)(a1 + 88) = v34;
          if ( *(_QWORD *)v26 != -4096 )
            --*(_DWORD *)(a1 + 92);
          *(_QWORD *)v26 = v35;
          *(_DWORD *)(v26 + 8) = v51;
          v36 = *(unsigned int *)(a1 + 112);
          v37 = *(unsigned int *)(a1 + 116);
          v52 = &v54;
          v55 = v22;
          v38 = v36 + 1;
          v53 = 0x800000000LL;
          v57 = 0x800000000LL;
          v30 = v36;
          v56 = v58;
          if ( v36 + 1 > v37 )
          {
            v42 = *(_QWORD *)(a1 + 104);
            if ( v42 > (unsigned __int64)&v55
              || (v43 = *(_QWORD *)(a1 + 104), v36 = v42 + 88 * v36, (unsigned __int64)&v55 >= v36) )
            {
              v47 = v26;
              sub_23590C0(v48, v38, v36, v26, (__int64)&v55, v20);
              v36 = *(unsigned int *)(a1 + 112);
              v39 = *(_QWORD *)(a1 + 104);
              v23 = (__int64)&v55;
              v26 = v47;
              v30 = v36;
            }
            else
            {
              v46 = v26;
              sub_23590C0(v48, v38, v36, v26, (__int64)&v55, v20);
              v39 = *(_QWORD *)(a1 + 104);
              v36 = *(unsigned int *)(a1 + 112);
              v26 = v46;
              v30 = v36;
              v23 = (__int64)&v55 + v39 - v43;
            }
          }
          else
          {
            v39 = *(_QWORD *)(a1 + 104);
          }
          v40 = (_QWORD *)(v39 + 88 * v36);
          if ( v40 )
          {
            *v40 = *(_QWORD *)v23;
            v40[1] = v40 + 3;
            v40[2] = 0x800000000LL;
            if ( *(_DWORD *)(v23 + 16) )
            {
              v45 = v26;
              sub_2B0B710((__int64)(v40 + 1), (char **)(v23 + 8), (__int64)v40, v26, v23, v20);
              v30 = *(unsigned int *)(a1 + 112);
              v26 = v45;
            }
            else
            {
              v30 = *(unsigned int *)(a1 + 112);
            }
          }
          v41 = v56;
          *(_DWORD *)(a1 + 112) = v30 + 1;
          if ( v41 != v58 )
          {
            v44 = v26;
            _libc_free((unsigned __int64)v41);
            v26 = v44;
            v30 = (unsigned int)(*(_DWORD *)(a1 + 112) - 1);
          }
          *(_DWORD *)(v26 + 8) = v30;
          goto LABEL_33;
        }
        sub_D39D40(v49, v21);
LABEL_63:
        sub_22B1A50(v49, (__int64 *)&v50, &v55);
        v35 = (__int64)v50;
        v26 = v55;
        v23 = (__int64)&v55;
        v34 = *(_DWORD *)(a1 + 88) + 1;
        goto LABEL_46;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 72);
      v55 = 0;
    }
    sub_D39D40(v49, 2 * v21);
    goto LABEL_63;
  }
}
