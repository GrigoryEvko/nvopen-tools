// Function: sub_17B79C0
// Address: 0x17b79c0
//
__int64 __fastcall sub_17B79C0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r13d
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // rsi
  _BYTE *v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // rdi
  unsigned __int64 v15; // rdx
  void *v16; // r9
  size_t v17; // r13
  int v18; // r8d
  char *v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // r8d
  int v23; // r9d
  const void *v24; // r15
  size_t v25; // rbx
  int v26; // eax
  char *v27; // rax
  char *v28; // rax
  _BYTE *v30; // r8
  __int64 v31; // rdx
  _BYTE *v32; // rsi
  char *v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdx
  _BYTE *v37; // rsi
  char *v38; // rdi
  char *v39; // rdi
  void *src; // [rsp+8h] [rbp-1F8h]
  int v43; // [rsp+18h] [rbp-1E8h]
  int v44; // [rsp+18h] [rbp-1E8h]
  _BYTE *v45; // [rsp+20h] [rbp-1E0h] BYREF
  __int64 v46; // [rsp+28h] [rbp-1D8h]
  _QWORD *v47; // [rsp+30h] [rbp-1D0h] BYREF
  __int16 v48; // [rsp+40h] [rbp-1C0h]
  char v49[16]; // [rsp+50h] [rbp-1B0h] BYREF
  __int16 v50; // [rsp+60h] [rbp-1A0h]
  char v51[16]; // [rsp+70h] [rbp-190h] BYREF
  __int16 v52; // [rsp+80h] [rbp-180h]
  char v53[16]; // [rsp+90h] [rbp-170h] BYREF
  __int16 v54; // [rsp+A0h] [rbp-160h]
  char *v55; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-148h]
  _BYTE v57[128]; // [rsp+C0h] [rbp-140h] BYREF
  char *p_dest; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+148h] [rbp-B8h]
  char dest; // [rsp+150h] [rbp-B0h] BYREF
  char v61; // [rsp+151h] [rbp-AFh]

  v61 = 1;
  p_dest = "llvm.gcov";
  dest = 3;
  v5 = sub_1632310(a2, (__int64)&p_dest);
  if ( !v5 || (v6 = v5, (v7 = sub_161F520(v5)) == 0) )
  {
LABEL_10:
    v13 = a3;
    if ( (*(_BYTE *)a3 == 15 || (v13 = *(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8))) != 0)
      && (v14 = *(_QWORD *)(v13 - 8LL * *(unsigned int *)(v13 + 8))) != 0 )
    {
      v16 = (void *)sub_161E970(v14);
      v55 = v57;
      v17 = v15;
      v18 = v15;
      v56 = 0x8000000000LL;
      if ( v15 <= 0x80 )
      {
        if ( !v15 )
          goto LABEL_35;
        v19 = v57;
      }
      else
      {
        src = v16;
        v43 = v15;
        sub_16CD150((__int64)&v55, v57, v15, 1, v15, (int)v16);
        v18 = v43;
        v16 = src;
        v19 = &v55[(unsigned int)v56];
      }
      v44 = v18;
      memcpy(v19, v16, v17);
      v18 = v44;
    }
    else
    {
      v18 = 0;
      v55 = v57;
      v56 = 0x8000000000LL;
    }
LABEL_35:
    LODWORD(v56) = v18 + v56;
    v33 = "gcno";
    v61 = 1;
    dest = 3;
    v34 = a1 + 16;
    if ( a4 )
      v33 = "gcda";
    p_dest = v33;
    sub_16C3CF0((__int64)&v55, (__int64)&p_dest, 2);
    v35 = sub_16C40A0((__int64)v55, (unsigned int)v56, 2);
    p_dest = &dest;
    v45 = (_BYTE *)v35;
    v46 = v36;
    v59 = 0x8000000000LL;
    if ( (unsigned int)sub_16C56A0(&p_dest) )
    {
      v37 = v45;
      *(_QWORD *)a1 = v34;
      if ( v37 )
      {
        sub_17B71F0((__int64 *)a1, v37, (__int64)&v37[v46]);
        v38 = p_dest;
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        v38 = p_dest;
        *(_BYTE *)(a1 + 16) = 0;
      }
    }
    else
    {
      v54 = 257;
      v52 = 257;
      v50 = 257;
      v48 = 261;
      v47 = &v45;
      sub_16C4D40((__int64)&p_dest, (__int64)&v47, (__int64)v49, (__int64)v51, (__int64)v53);
      v38 = p_dest;
      *(_QWORD *)a1 = v34;
      if ( v38 )
      {
        sub_17B71F0((__int64 *)a1, v38, (__int64)&v38[(unsigned int)v59]);
        v38 = p_dest;
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        *(_BYTE *)(a1 + 16) = 0;
      }
    }
    if ( v38 != &dest )
      _libc_free((unsigned __int64)v38);
    if ( v55 != v57 )
      _libc_free((unsigned __int64)v55);
    return a1;
  }
  v8 = 0;
  while ( 1 )
  {
    v9 = sub_161F530(v6, v8);
    v10 = *(_DWORD *)(v9 + 8);
    if ( v10 == 3 )
    {
      v11 = 2;
    }
    else
    {
      v11 = 1;
      if ( v10 != 2 )
        goto LABEL_9;
    }
    v12 = *(_BYTE **)(v9 + 8 * (v11 - v10));
    if ( (unsigned __int8)(*v12 - 4) >= 0x1Fu )
      v12 = 0;
    if ( (_BYTE *)a3 != v12 )
      goto LABEL_9;
    if ( v10 != 3 )
      break;
    v30 = *(_BYTE **)(v9 - 24);
    if ( !*v30 && !**(_BYTE **)(v9 - 16) )
    {
      if ( a4 )
        v32 = (_BYTE *)sub_161E970(*(_QWORD *)(v9 - 16));
      else
        v32 = (_BYTE *)sub_161E970((__int64)v30);
      *(_QWORD *)a1 = a1 + 16;
      if ( v32 )
      {
        sub_17B71F0((__int64 *)a1, v32, (__int64)&v32[v31]);
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        *(_BYTE *)(a1 + 16) = 0;
      }
      return a1;
    }
LABEL_9:
    if ( v7 == ++v8 )
      goto LABEL_10;
  }
  if ( **(_BYTE **)(v9 - 8LL * v10) )
    goto LABEL_9;
  v20 = sub_161E970(*(_QWORD *)(v9 - 8LL * v10));
  p_dest = &dest;
  v24 = (const void *)v20;
  v59 = 0x8000000000LL;
  v25 = v21;
  v26 = v21;
  if ( v21 > 0x80 )
  {
    sub_16CD150((__int64)&p_dest, &dest, v21, 1, v22, v23);
    v39 = &p_dest[(unsigned int)v59];
  }
  else
  {
    if ( !v21 )
      goto LABEL_21;
    v39 = &dest;
  }
  memcpy(v39, v24, v25);
  v26 = v25 + v59;
LABEL_21:
  LODWORD(v59) = v26;
  v27 = "gcno";
  v57[1] = 1;
  v57[0] = 3;
  if ( a4 )
    v27 = "gcda";
  v55 = v27;
  sub_16C3CF0((__int64)&p_dest, (__int64)&v55, 2);
  v28 = p_dest;
  *(_QWORD *)a1 = a1 + 16;
  if ( v28 )
  {
    sub_17B71F0((__int64 *)a1, v28, (__int64)&v28[(unsigned int)v59]);
    v28 = p_dest;
    if ( p_dest == &dest )
      return a1;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  _libc_free((unsigned __int64)v28);
  return a1;
}
