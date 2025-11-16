// Function: sub_269C5D0
// Address: 0x269c5d0
//
void __fastcall sub_269C5D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 v11; // rsi
  __int64 *v12; // rbx
  __int64 *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  _QWORD *v16; // rdx
  __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r14
  unsigned __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 *v25; // rax
  __int64 *v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  bool v32; // zf
  _QWORD *v33; // rcx
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rcx
  _QWORD *v38; // rdx
  __int64 v39; // [rsp+20h] [rbp-170h] BYREF
  unsigned __int64 v40; // [rsp+28h] [rbp-168h] BYREF
  _QWORD *v41; // [rsp+30h] [rbp-160h]
  _QWORD *v42; // [rsp+38h] [rbp-158h]
  __int64 v43; // [rsp+40h] [rbp-150h]
  __int64 v44; // [rsp+48h] [rbp-148h]
  _QWORD *v45; // [rsp+50h] [rbp-140h]
  _QWORD *i; // [rsp+58h] [rbp-138h]
  _QWORD *v47; // [rsp+60h] [rbp-130h]
  __int64 v48; // [rsp+68h] [rbp-128h]
  _BYTE *v49; // [rsp+70h] [rbp-120h] BYREF
  __int64 v50; // [rsp+78h] [rbp-118h]
  _BYTE v51[48]; // [rsp+80h] [rbp-110h] BYREF
  unsigned __int64 v52; // [rsp+B0h] [rbp-E0h] BYREF
  _QWORD *v53; // [rsp+B8h] [rbp-D8h]
  _QWORD *v54; // [rsp+C0h] [rbp-D0h]
  __int64 v55; // [rsp+C8h] [rbp-C8h]
  _BYTE *v56; // [rsp+D0h] [rbp-C0h]
  __int64 v57; // [rsp+D8h] [rbp-B8h]
  _BYTE v58[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v2 = a2;
  v4 = *a1;
  if ( a2 )
  {
    v52 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v5 = sub_26735F0(v4 + 256, &v52);
  }
  else
  {
    v52 = 0;
    v5 = sub_26868A0(v4 + 224, &v52);
  }
  if ( !*((_BYTE *)v5 + 1)
    || *((_BYTE *)v5 + 3)
    || *((_DWORD *)v5 + 20) != *((_DWORD *)v5 + 19) && !*(_BYTE *)(a1[1] + 4296) )
  {
    return;
  }
  if ( a2 )
  {
    sub_AE6EC0(a1[2], a2);
    sub_2570110(a1[1], a2);
    *(_DWORD *)a1[3] = 0;
    v8 = *((unsigned int *)v5 + 19);
    if ( (_DWORD)v8 == *((_DWORD *)v5 + 20) )
      return;
    goto LABEL_9;
  }
  if ( *((_DWORD *)v5 + 7) == *((_DWORD *)v5 + 8) )
    return;
  *(_DWORD *)a1[3] = 0;
  v14 = (_QWORD *)v5[2];
  v15 = v5[1];
  if ( *((_BYTE *)v5 + 36) )
    v16 = &v14[*((unsigned int *)v5 + 7)];
  else
    v16 = &v14[*((unsigned int *)v5 + 6)];
  while ( v16 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
    ++v14;
  v45 = v14;
  v47 = v5 + 1;
  v48 = v15;
  v54 = v5 + 1;
  v55 = v15;
  i = v16;
  v52 = (unsigned __int64)v16;
  v53 = v16;
  v49 = v51;
  v50 = 0x600000000LL;
  sub_2677A40((__int64)&v49, (__int64)(v5 + 1), (__int64)v16, v15, v6, v7, v14, v16, (_DWORD)v5 + 8, v15, v16);
  v17 = 0;
  v56 = v58;
  v57 = 0x1000000000LL;
  v18 = v50;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  if ( !(_DWORD)v50 )
    goto LABEL_44;
  do
  {
    v19 = *(_QWORD *)&v49[8 * v18 - 8];
    LODWORD(v50) = v18 - 1;
    v39 = v19;
    if ( (unsigned __int8)sub_269C270((__int64)&v52, &v39) )
    {
      v20 = sub_B43CB0(v39);
      if ( v20 == sub_25096F0((_QWORD *)(*a1 + 72)) )
      {
        v21 = *(_QWORD *)(v39 + 40);
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v22 == v21 + 48 )
            break;
          if ( !v22 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 > 0xA || !(unsigned int)sub_B46E30(v22 - 24) )
            break;
          v21 = sub_AA5780(v21);
          if ( !v21 )
            goto LABEL_29;
        }
        v23 = a1[2];
        v24 = v39;
        if ( *(_BYTE *)(v23 + 28) )
        {
          v25 = *(__int64 **)(v23 + 8);
          v26 = &v25[*(unsigned int *)(v23 + 20)];
          if ( v25 == v26 )
            goto LABEL_62;
          while ( 1 )
          {
            v27 = *v25;
            if ( v39 == *v25 )
              break;
            if ( v26 == ++v25 )
              goto LABEL_62;
          }
        }
        else
        {
          if ( !sub_C8CA60(v23, v39) )
          {
            v24 = v39;
LABEL_62:
            sub_2570110(a1[1], v24);
            goto LABEL_29;
          }
          v27 = v39;
        }
        v28 = *a1;
        v40 = v27 & 0xFFFFFFFFFFFFFFFBLL;
        v29 = sub_26735F0(v28 + 256, &v40);
        v32 = *((_BYTE *)v29 + 36) == 0;
        v33 = v29;
        v34 = (_QWORD *)v29[2];
        if ( v32 )
          v35 = *((unsigned int *)v33 + 6);
        else
          v35 = *((unsigned int *)v33 + 7);
        v36 = (__int64)(v33 + 1);
        v37 = v33[1];
        v38 = &v34[v35];
        v41 = v38;
        v42 = v38;
        v43 = v36;
        v44 = v37;
        for ( i = v38; v38 != v34; ++v34 )
        {
          if ( *v34 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
        v47 = (_QWORD *)v36;
        v48 = v37;
        v45 = v34;
        sub_2677A40((__int64)&v49, v36, (__int64)v38, v37, v30, v31, v34, i, v36, v37, v41);
      }
    }
LABEL_29:
    v18 = v50;
  }
  while ( (_DWORD)v50 );
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  v2 = (__int64)v53;
  v17 = 8LL * (unsigned int)v55;
LABEL_44:
  sub_C7D6A0(v2, v17, 8);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  v8 = *((unsigned int *)v5 + 19);
  if ( *((_DWORD *)v5 + 20) != (_DWORD)v8 && *((_DWORD *)v5 + 7) != *((_DWORD *)v5 + 8) )
  {
LABEL_9:
    v9 = (__int64 *)v5[8];
    if ( !*((_BYTE *)v5 + 84) )
      v8 = *((unsigned int *)v5 + 18);
    v10 = &v9[v8];
    if ( v9 != v10 )
    {
      while ( 1 )
      {
        v11 = *v9;
        v12 = v9;
        if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == ++v9 )
          return;
      }
      while ( v10 != v12 )
      {
        sub_2570110(a1[1], v11);
        v13 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        while ( 1 )
        {
          v11 = *v13;
          v12 = v13;
          if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v13 )
            return;
        }
      }
    }
  }
}
