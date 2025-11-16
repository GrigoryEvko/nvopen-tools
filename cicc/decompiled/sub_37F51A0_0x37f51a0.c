// Function: sub_37F51A0
// Address: 0x37f51a0
//
__int64 __fastcall sub_37F51A0(__int64 a1, __int64 a2)
{
  __int64 v3; // r10
  __int64 v4; // r11
  __int64 v5; // rax
  int v6; // r8d
  __int16 v7; // dx
  __int64 result; // rax
  __int64 v9; // r11
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 v12; // r9
  int v13; // ebx
  _DWORD *v14; // r14
  int v15; // ebx
  _QWORD *v16; // rcx
  __int64 *v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // r15
  __int64 *v20; // rdx
  int v21; // edx
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  char *v26; // r15
  _QWORD *v27; // rsi
  char *v28; // rdx
  char *v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 *v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // rdi
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  __int64 v44; // [rsp+18h] [rbp-88h]
  __int64 v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  unsigned __int64 v48; // [rsp+20h] [rbp-80h]
  __int64 *v49; // [rsp+20h] [rbp-80h]
  unsigned __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned __int64 v51; // [rsp+20h] [rbp-80h]
  __int64 v52; // [rsp+20h] [rbp-80h]
  int v53; // [rsp+2Ch] [rbp-74h]
  __int64 v54; // [rsp+30h] [rbp-70h]
  __int64 *v55; // [rsp+38h] [rbp-68h]
  _QWORD v56[2]; // [rsp+40h] [rbp-60h] BYREF
  char v57; // [rsp+50h] [rbp-50h]
  __int64 v58; // [rsp+58h] [rbp-48h]

  sub_37F4460((__int64)v56, *(_QWORD *)(a2 + 56), a2 + 48, 1);
  v5 = v56[0];
  if ( v56[0] == v58 )
  {
    v53 = 0;
  }
  else
  {
    v6 = 0;
    do
    {
      do
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v56[1] )
          break;
        v7 = *(_WORD *)(v5 + 68);
      }
      while ( (unsigned __int16)(v7 - 14) <= 4u || v57 == 1 && v7 == 24 );
      ++v6;
    }
    while ( v5 != v58 );
    v53 = v6;
  }
  result = *(_QWORD *)(v4 + 64);
  v39 = result + 8LL * *(unsigned int *)(v4 + 72);
  if ( v39 != result )
  {
    v54 = *(_QWORD *)(v4 + 64);
    v9 = 24 * v3;
    while ( 1 )
    {
      v55 = (__int64 *)(*(_QWORD *)(a1 + 344) + 24LL * *(int *)(*(_QWORD *)v54 + 24LL));
      v10 = *v55;
      if ( *v55 != v55[1] )
      {
        if ( *(_DWORD *)(a1 + 304) )
          break;
      }
LABEL_46:
      v54 += 8;
      result = v54;
      if ( v39 == v54 )
        return result;
    }
    v11 = 0;
    while ( 1 )
    {
      v15 = *(_DWORD *)(v10 + 4LL * v11);
      if ( *(_DWORD *)(a1 + 640) == v15 )
        goto LABEL_18;
      v16 = (_QWORD *)(v9 + *(_QWORD *)(a1 + 496));
      v17 = (__int64 *)(*v16 + 8LL * v11);
      v18 = *v17;
      v19 = *v17 & 0xFFFFFFFFFFFFFFFELL;
      if ( v16[1] == *v16 )
        break;
      if ( !v19 )
      {
        v12 = 4LL * v15 + 2;
        if ( v18 == 1 )
        {
          v23 = 0;
          v24 = 8LL * MEMORY[8];
          if ( v24 )
          {
            v25 = MEMORY[8] + 1LL;
            if ( v25 > MEMORY[0xC] )
              goto LABEL_50;
LABEL_29:
            v26 = *(char **)v23;
            v27 = (_QWORD *)(*(_QWORD *)v23 + v24);
            v28 = (char *)(v27 - 1);
LABEL_30:
            *v27 = *(_QWORD *)v28;
            v29 = *(char **)v23;
            v30 = *(unsigned int *)(v23 + 8);
            v31 = 8 * v30;
            v28 = (char *)(*(_QWORD *)v23 + 8 * v30 - 8);
LABEL_31:
            if ( v28 != v26 )
            {
              v48 = v23;
              v40 = v9;
              v43 = v12;
              memmove(&v29[v31 - (v28 - v26)], v26, v28 - v26);
              v23 = v48;
              v9 = v40;
              v12 = v43;
              LODWORD(v30) = *(_DWORD *)(v48 + 8);
            }
            *(_DWORD *)(v23 + 8) = v30 + 1;
            *(_QWORD *)v26 = v12;
            goto LABEL_16;
          }
        }
        goto LABEL_15;
      }
      v20 = (__int64 *)(*v16 + 8LL * v11);
      if ( (v18 & 1) != 0 )
      {
        v20 = *(__int64 **)v19;
        if ( !*(_DWORD *)(v19 + 8) )
        {
          v12 = 4LL * v15 + 2;
          goto LABEL_27;
        }
      }
      v21 = (int)*v20 >> 2;
      if ( v21 >= 0 )
      {
        v12 = 4LL * v15 + 2;
        if ( (v18 & 1) == 0 )
          goto LABEL_35;
LABEL_27:
        v22 = *(unsigned int *)(v19 + 8);
        v23 = *v17 & 0xFFFFFFFFFFFFFFFELL;
        v24 = 8 * v22;
        if ( !(8 * v22) )
          goto LABEL_55;
LABEL_28:
        v25 = v22 + 1;
        if ( v25 <= *(unsigned int *)(v23 + 12) )
          goto LABEL_29;
LABEL_50:
        v50 = v23;
        v42 = v9;
        v45 = v12;
        sub_C8D5F0(v23, (const void *)(v23 + 16), v25, 8u, (__int64)v17, v12);
        v23 = v50;
        v12 = v45;
        v9 = v42;
        v29 = *(char **)v50;
        v30 = *(unsigned int *)(v50 + 8);
        v31 = 8 * v30;
        v26 = *(char **)v50;
        v27 = (_QWORD *)(8 * v30 + *(_QWORD *)v50);
        v28 = (char *)(v27 - 1);
        if ( v27 )
          goto LABEL_30;
        goto LABEL_31;
      }
      if ( v15 <= v21 )
        goto LABEL_18;
      if ( (v18 & 1) != 0 )
        v17 = *(__int64 **)v19;
      *v17 = 4LL * v15 + 2;
LABEL_16:
      v13 = v15 - v53;
      v14 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 344) + v9) + 4LL * v11);
      if ( *v14 < v13 )
        *v14 = v13;
LABEL_18:
      if ( *(_DWORD *)(a1 + 304) == ++v11 )
        goto LABEL_46;
      v10 = *v55;
    }
    v12 = 4LL * v15 + 2;
    if ( (v18 & 1) != 0 )
    {
      v22 = *(unsigned int *)(v19 + 8);
      v23 = *v17 & 0xFFFFFFFFFFFFFFFELL;
      v24 = 8 * v22;
      if ( 8 * v22 )
        goto LABEL_28;
      if ( v19 )
      {
LABEL_55:
        if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v19 + 12) )
        {
          v47 = v9;
          v52 = v12;
          sub_C8D5F0(v19, (const void *)(v19 + 16), v22 + 1, 8u, v22 + 1, v12);
          v22 = *(unsigned int *)(v19 + 8);
          v9 = v47;
          v12 = v52;
        }
        *(_QWORD *)(*(_QWORD *)v19 + 8 * v22) = v12;
        ++*(_DWORD *)(v19 + 8);
        goto LABEL_16;
      }
    }
    else if ( v19 )
    {
LABEL_35:
      *v17 = v12;
      v41 = v9;
      v44 = v12;
      v49 = v17;
      v32 = sub_22077B0(0x30u);
      v33 = v49;
      v34 = v44;
      v9 = v41;
      if ( v32 )
      {
        *(_QWORD *)v32 = v32 + 16;
        *(_QWORD *)(v32 + 8) = 0x400000000LL;
      }
      v35 = v32 & 0xFFFFFFFFFFFFFFFELL;
      *v49 = v32 | 1;
      v36 = *(unsigned int *)((v32 & 0xFFFFFFFFFFFFFFFELL) + 8);
      if ( v36 + 1 > (unsigned __int64)*(unsigned int *)((v32 & 0xFFFFFFFFFFFFFFFELL) + 12) )
      {
        sub_C8D5F0(v35, (const void *)(v35 + 16), v36 + 1, 8u, (__int64)v49, v44);
        v9 = v41;
        v33 = v49;
        v34 = v44;
        v36 = *(unsigned int *)(v35 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v35 + 8 * v36) = v34;
      ++*(_DWORD *)(v35 + 8);
      v37 = *v33 & 0xFFFFFFFFFFFFFFFELL;
      v38 = *(unsigned int *)(v37 + 8);
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 12) )
      {
        v46 = v9;
        v51 = *v33 & 0xFFFFFFFFFFFFFFFELL;
        sub_C8D5F0(v37, (const void *)(v37 + 16), v38 + 1, 8u, v38 + 1, v34);
        v37 = v51;
        v9 = v46;
        v38 = *(unsigned int *)(v51 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v37 + 8 * v38) = v19;
      ++*(_DWORD *)(v37 + 8);
      goto LABEL_16;
    }
LABEL_15:
    *v17 = v12;
    goto LABEL_16;
  }
  return result;
}
