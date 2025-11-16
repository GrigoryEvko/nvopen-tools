// Function: sub_37BB480
// Address: 0x37bb480
//
__int64 __fastcall sub_37BB480(__int64 a1, unsigned int a2, unsigned int a3)
{
  char *v4; // rax
  __int64 v5; // rdx
  unsigned __int16 *v6; // r15
  char *i; // r14
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r8
  __int64 v11; // rdi
  int v12; // ebx
  int v13; // r8d
  __int64 v14; // rax
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 v17; // r15
  __int64 v18; // rdx
  unsigned int *v19; // r14
  unsigned int v20; // eax
  __int64 v21; // r14
  __int64 v22; // r15
  unsigned int *v23; // rdx
  unsigned int v24; // eax
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  __int16 *v27; // rax
  __int16 *v28; // rbx
  __int64 result; // rax
  int v30; // r15d
  unsigned __int16 v31; // r14
  unsigned __int16 *v32; // r13
  unsigned int v33; // r8d
  __int64 v34; // rdi
  __int64 v35; // rax
  unsigned int v36; // r9d
  __int64 v37; // r10
  __int64 v38; // rsi
  _DWORD *v39; // r14
  __int64 v40; // r14
  _DWORD *v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // r11
  __int64 v44; // r14
  unsigned int v45; // eax
  int v46; // eax
  __int64 v47; // rsi
  int v48; // eax
  unsigned int *v49; // r10
  _DWORD *v51; // [rsp+8h] [rbp-58h]
  unsigned int *v52; // [rsp+10h] [rbp-50h]
  __int64 v53; // [rsp+10h] [rbp-50h]
  __int64 v54; // [rsp+10h] [rbp-50h]
  unsigned int v55; // [rsp+10h] [rbp-50h]
  int v56; // [rsp+18h] [rbp-48h]
  unsigned int *v57; // [rsp+18h] [rbp-48h]
  unsigned int v58; // [rsp+18h] [rbp-48h]
  unsigned int v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  unsigned int v61; // [rsp+20h] [rbp-40h]
  unsigned int v62; // [rsp+20h] [rbp-40h]
  unsigned int *v63; // [rsp+20h] [rbp-40h]
  unsigned int *v64; // [rsp+20h] [rbp-40h]

  v4 = sub_E922F0(*(_QWORD **)(a1 + 16), a3);
  v6 = (unsigned __int16 *)&v4[2 * v5];
  for ( i = v4; v6 != (unsigned __int16 *)i; *(_DWORD *)(v8 + 4) = (unsigned __int8)((v10 | v9) >> 32) | (v15 << 8) )
  {
    v11 = *(_QWORD *)(a1 + 408);
    v12 = *(_DWORD *)(a1 + 420);
    v13 = *(_DWORD *)(a1 + 416);
    v14 = *(unsigned __int16 *)i;
    v15 = *(_DWORD *)(*(_QWORD *)(v11 + 64) + 4 * v14);
    if ( v15 == -1 )
    {
      v52 = (unsigned int *)(*(_QWORD *)(v11 + 64) + 4 * v14);
      v56 = *(_DWORD *)(a1 + 416);
      v16 = sub_37BA230(v11, (unsigned __int16)v14);
      v13 = v56;
      v15 = v16;
      *v52 = v16;
    }
    i += 2;
    v8 = *(_QWORD *)(v11 + 32) + 8LL * v15;
    v9 = v13 & 0xFFFFF | ((unsigned __int64)(v12 & 0xFFFFF) << 20);
    v10 = *(_QWORD *)v8 & 0xFFFFFF0000000000LL;
    *(_QWORD *)v8 = v10 | v9;
  }
  v17 = *(_QWORD *)(a1 + 408);
  v18 = *(_QWORD *)(v17 + 64);
  v19 = (unsigned int *)(v18 + 4LL * a2);
  v20 = *v19;
  if ( *v19 == -1 )
  {
    v20 = sub_37BA230(*(_QWORD *)(a1 + 408), a2);
    *v19 = v20;
    v21 = *(_QWORD *)(a1 + 408);
    v18 = *(_QWORD *)(v21 + 64);
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 408);
  }
  v22 = *(_QWORD *)(*(_QWORD *)(v17 + 32) + 8LL * v20);
  v23 = (unsigned int *)(v18 + 4LL * a3);
  v24 = *v23;
  if ( *v23 == -1 )
  {
    v64 = v23;
    v24 = sub_37BA230(v21, a3);
    *v64 = v24;
  }
  *(_QWORD *)(*(_QWORD *)(v21 + 32) + 8LL * v24) = v22;
  v25 = *(_QWORD **)(a1 + 16);
  v26 = v25[1] + 24LL * a2;
  v27 = (__int16 *)(v25[7] + 2LL * *(unsigned int *)(v26 + 4));
  v28 = v27 + 1;
  result = (unsigned int)*v27;
  v30 = a2 + result;
  if ( (_WORD)result )
  {
    v31 = a2 + result;
    v32 = (unsigned __int16 *)(v25[11] + 2LL * *(unsigned int *)(v26 + 12));
    while ( 1 )
    {
      v33 = sub_E91CF0(v25, a3, *v32);
      if ( !v33 )
        goto LABEL_19;
      v34 = *(_QWORD *)(a1 + 408);
      v35 = v31;
      v36 = v31;
      v37 = 4LL * v31;
      v38 = *(_QWORD *)(v34 + 64);
      v39 = (_DWORD *)(v38 + v37);
      if ( *(_DWORD *)(v38 + v37) == -1 )
      {
        v53 = v37;
        v58 = v33;
        v61 = v36;
        v46 = sub_37BA230(v34, v36);
        v36 = v61;
        v37 = v53;
        *v39 = v46;
        v43 = *(_QWORD *)(a1 + 408);
        v47 = *(_QWORD *)(v43 + 64);
        v33 = v58;
        v40 = 4LL * v58;
        v41 = (_DWORD *)(v47 + v40);
        if ( *(_DWORD *)(v47 + v40) != -1 )
          goto LABEL_25;
        v34 = *(_QWORD *)(a1 + 408);
      }
      else
      {
        v40 = 4LL * v33;
        v41 = (_DWORD *)(v38 + v40);
        if ( *(_DWORD *)(v38 + v40) != -1 )
        {
          v42 = *(_DWORD *)(v38 + 4 * v35);
          v43 = *(_QWORD *)(a1 + 408);
          goto LABEL_16;
        }
      }
      v51 = v41;
      v54 = v37;
      v59 = v36;
      v62 = v33;
      v48 = sub_37BA230(v34, v33);
      v37 = v54;
      v36 = v59;
      v33 = v62;
      *v51 = v48;
      v43 = *(_QWORD *)(a1 + 408);
      v47 = *(_QWORD *)(v43 + 64);
LABEL_25:
      v49 = (unsigned int *)(v47 + v37);
      v42 = *v49;
      v63 = v49;
      if ( *v49 == -1 )
      {
        v55 = v33;
        v60 = v43;
        v42 = sub_37BA230(v43, v36);
        v43 = v60;
        v33 = v55;
        *v63 = v42;
        v34 = *(_QWORD *)(a1 + 408);
        v41 = (_DWORD *)(v40 + *(_QWORD *)(v34 + 64));
      }
      else
      {
        v34 = v43;
        v41 = (_DWORD *)(v47 + v40);
      }
LABEL_16:
      v44 = *(_QWORD *)(*(_QWORD *)(v43 + 32) + 8LL * v42);
      v45 = *v41;
      if ( *v41 == -1 )
      {
        v57 = v41;
        v45 = sub_37BA230(v34, v33);
        *v57 = v45;
      }
      *(_QWORD *)(*(_QWORD *)(v34 + 32) + 8LL * v45) = v44;
LABEL_19:
      result = (unsigned int)*v28;
      ++v32;
      ++v28;
      if ( !(_WORD)result )
        return result;
      v30 += result;
      v25 = *(_QWORD **)(a1 + 16);
      v31 = v30;
    }
  }
  return result;
}
