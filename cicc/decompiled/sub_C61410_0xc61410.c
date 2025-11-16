// Function: sub_C61410
// Address: 0xc61410
//
_BYTE *__fastcall sub_C61410(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  void *v6; // rdi
  unsigned __int64 v7; // r13
  const void *v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // r12
  _BYTE *result; // rax
  __int64 i; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rdx
  int v21; // eax
  int v22; // edx
  __int64 v23; // rsi
  int v24; // ecx
  unsigned int v25; // edx
  int *v26; // r15
  int v27; // edi
  _BYTE *v28; // rsi
  __int64 v29; // rdx
  char v30; // al
  unsigned int v31; // r9d
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdi
  unsigned int v37; // r15d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdi
  int v41; // r9d
  char *v42; // r10
  size_t v43; // rdx
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-138h]
  __int64 v46; // [rsp+8h] [rbp-138h]
  __int64 v47; // [rsp+28h] [rbp-118h]
  unsigned int v48; // [rsp+28h] [rbp-118h]
  unsigned int v49; // [rsp+28h] [rbp-118h]
  _BYTE *v50; // [rsp+30h] [rbp-110h]
  _QWORD *v52; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-E8h]
  _QWORD v54[2]; // [rsp+60h] [rbp-E0h] BYREF
  _OWORD *v55; // [rsp+70h] [rbp-D0h]
  __int64 v56; // [rsp+78h] [rbp-C8h]
  _OWORD v57[12]; // [rsp+80h] [rbp-C0h] BYREF

  v3 = sub_CB7210(a1);
  v4 = *(_QWORD *)(v3 + 32);
  v5 = v3;
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 2 )
  {
    v44 = sub_CB6200(v3, "  -", 3);
    v6 = *(void **)(v44 + 32);
    v5 = v44;
  }
  else
  {
    *(_BYTE *)(v4 + 2) = 45;
    *(_WORD *)v4 = 8224;
    v6 = (void *)(*(_QWORD *)(v3 + 32) + 3LL);
    *(_QWORD *)(v3 + 32) = v6;
  }
  v7 = *(_QWORD *)(a1 + 32);
  v8 = *(const void **)(a1 + 24);
  if ( v7 > *(_QWORD *)(v5 + 24) - (_QWORD)v6 )
  {
    sub_CB6200(v5, v8, *(_QWORD *)(a1 + 32));
    v7 = *(_QWORD *)(a1 + 32);
  }
  else if ( v7 )
  {
    memcpy(v6, v8, *(_QWORD *)(a1 + 32));
    *(_QWORD *)(v5 + 32) += v7;
    v7 = *(_QWORD *)(a1 + 32);
  }
  sub_C540D0(*(_OWORD *)(a1 + 40), a2, v7 + 6);
  v9 = sub_C60B10();
  v10 = v9[10];
  v11 = v9;
  result = (_BYTE *)v9[11];
  v50 = result;
  if ( (_BYTE *)v10 != result )
  {
    for ( i = sub_C61310((__int64)(v11 + 4), v10); ; i = sub_C61310((__int64)(v11 + 4), v10) )
    {
      v19 = 0x1FFFFFFFE0LL;
      v20 = i;
      v21 = 0;
      if ( (__int64 *)v20 != v11 + 5 )
      {
        v21 = *(_DWORD *)(v20 + 64);
        v19 = 32LL * (unsigned int)(v21 - 1);
      }
      v22 = *((_DWORD *)v11 + 6);
      v23 = v11[1];
      if ( !v22 )
        goto LABEL_33;
      v24 = v22 - 1;
      v25 = (v22 - 1) & (37 * v21);
      v26 = (int *)(v23 + ((unsigned __int64)v25 << 7));
      v27 = *v26;
      if ( v21 != *v26 )
        break;
LABEL_23:
      v28 = (_BYTE *)*((_QWORD *)v26 + 4);
      v47 = v19;
      v29 = *((_QWORD *)v26 + 5);
      v57[1] = *(_OWORD *)(v26 + 2);
      v30 = *((_BYTE *)v26 + 24);
      *((_QWORD *)&v57[2] + 1) = (char *)&v57[3] + 8;
      LOBYTE(v57[2]) = v30;
      sub_C5FA40((__int64 *)&v57[2] + 1, v28, (__int64)&v28[v29]);
      v31 = v26[18];
      *((_QWORD *)&v57[4] + 1) = (char *)&v57[5] + 8;
      *(_QWORD *)&v57[5] = 0x300000000LL;
      v19 = v47;
      if ( v31 && (int *)((char *)&v57[4] + 8) != v26 + 16 )
      {
        v42 = (char *)&v57[5] + 8;
        v43 = 16LL * v31;
        if ( v31 <= 3
          || (v46 = v47,
              v49 = v31,
              sub_C8D5F0((char *)&v57[4] + 8, (char *)&v57[5] + 8, v31, 16),
              v42 = (char *)*((_QWORD *)&v57[4] + 1),
              v31 = v49,
              v19 = v46,
              (v43 = 16LL * (unsigned int)v26[18]) != 0) )
        {
          v45 = v19;
          v48 = v31;
          memcpy(v42, *((const void **)v26 + 8), v43);
          v19 = v45;
          LODWORD(v57[5]) = v48;
        }
        else
        {
          LODWORD(v57[5]) = v49;
        }
      }
LABEL_24:
      v32 = v11[10] + v19;
      v33 = *(_QWORD *)(v32 + 8);
      v52 = v54;
      v34 = *(_QWORD *)v32;
      sub_C5FA40((__int64 *)&v52, *(_BYTE **)v32, *(_QWORD *)v32 + v33);
      v55 = v57;
      if ( *((_OWORD **)&v57[2] + 1) == (_OWORD *)((char *)&v57[3] + 8) )
      {
        v57[0] = _mm_loadu_si128((const __m128i *)((char *)&v57[3] + 8));
      }
      else
      {
        v55 = (_OWORD *)*((_QWORD *)&v57[2] + 1);
        *(_QWORD *)&v57[0] = *((_QWORD *)&v57[3] + 1);
      }
      v35 = *(_QWORD *)&v57[3];
      v36 = *((_QWORD *)&v57[4] + 1);
      *((_QWORD *)&v57[2] + 1) = (char *)&v57[3] + 8;
      *(_QWORD *)&v57[3] = 0;
      v56 = v35;
      BYTE8(v57[3]) = 0;
      if ( *((_OWORD **)&v57[4] + 1) != (_OWORD *)((char *)&v57[5] + 8) )
      {
        _libc_free(*((_QWORD *)&v57[4] + 1), v34);
        v36 = *((_QWORD *)&v57[2] + 1);
        if ( *((_OWORD **)&v57[2] + 1) != (_OWORD *)((char *)&v57[3] + 8) )
          j_j___libc_free_0(*((_QWORD *)&v57[2] + 1), *((_QWORD *)&v57[3] + 1) + 1LL);
      }
      v37 = a2 - v53 - 8;
      v38 = sub_CB7210(v36);
      v39 = *(_QWORD *)(v38 + 32);
      v40 = v38;
      if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v39) > 4 )
      {
        *(_DWORD *)v39 = 538976288;
        *(_BYTE *)(v39 + 4) = 61;
        *(_QWORD *)(v38 + 32) += 5LL;
      }
      else
      {
        v40 = sub_CB6200(v38, "    =", 5);
      }
      sub_CB6200(v40, v52, v53);
      v14 = sub_CB7210(v40);
      v15 = sub_CB69B0(v14, v37);
      v16 = *(_QWORD *)(v15 + 32);
      v17 = v15;
      if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 4 )
      {
        v17 = sub_CB6200(v15, " -   ", 5);
      }
      else
      {
        *(_DWORD *)v16 = 538979616;
        *(_BYTE *)(v16 + 4) = 32;
        *(_QWORD *)(v15 + 32) += 5LL;
      }
      v18 = sub_CB6200(v17, v55, v56);
      result = *(_BYTE **)(v18 + 32);
      if ( (unsigned __int64)result >= *(_QWORD *)(v18 + 24) )
      {
        result = (_BYTE *)sub_CB5D20(v18, 10);
      }
      else
      {
        *(_QWORD *)(v18 + 32) = result + 1;
        *result = 10;
      }
      if ( v55 != v57 )
        result = (_BYTE *)j_j___libc_free_0(v55, *(_QWORD *)&v57[0] + 1LL);
      if ( v52 != v54 )
        result = (_BYTE *)j_j___libc_free_0(v52, v54[0] + 1LL);
      v10 += 32;
      if ( v50 == (_BYTE *)v10 )
        return result;
    }
    v41 = 1;
    while ( v27 != -1 )
    {
      v25 = v24 & (v41 + v25);
      v26 = (int *)(v23 + ((unsigned __int64)v25 << 7));
      v27 = *v26;
      if ( v21 == *v26 )
        goto LABEL_23;
      ++v41;
    }
LABEL_33:
    memset(&v57[1], 0, 0x78u);
    *((_QWORD *)&v57[2] + 1) = (char *)&v57[3] + 8;
    *((_QWORD *)&v57[4] + 1) = (char *)&v57[5] + 8;
    DWORD1(v57[5]) = 3;
    goto LABEL_24;
  }
  return result;
}
