// Function: sub_1310E90
// Address: 0x1310e90
//
unsigned __int16 __fastcall sub_1310E90(__int64 a1, __int64 *a2, char **a3, unsigned int a4, int a5)
{
  __int64 v5; // r15
  __int64 v7; // r12
  __int64 v9; // rdi
  __int16 v10; // ax
  unsigned int v11; // ebx
  pthread_mutex_t *v12; // r13
  void *v13; // rsp
  _QWORD **v14; // r14
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r12
  char v19; // al
  char **v20; // rcx
  __int64 v21; // r13
  _QWORD *v22; // rsi
  pthread_mutex_t *v23; // rax
  _QWORD **v24; // rax
  __int64 v25; // r14
  __int64 v26; // r13
  _QWORD **v27; // r15
  _QWORD *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdi
  int v31; // ecx
  _QWORD **v32; // rax
  int v33; // ebx
  bool v34; // sf
  __int64 v35; // r13
  pthread_mutex_t *v36; // rax
  __int64 *v37; // r14
  char *v38; // rsi
  __int64 v39; // rbx
  __int64 v40; // rdx
  unsigned __int16 v41; // cx
  unsigned __int16 result; // ax
  char **v43; // rdi
  _QWORD *v44; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 *v45; // [rsp+8h] [rbp-88h]
  _QWORD **v46; // [rsp+10h] [rbp-80h]
  volatile signed __int64 *v47; // [rsp+18h] [rbp-78h]
  volatile signed __int64 *v48; // [rsp+20h] [rbp-70h]
  char **v49; // [rsp+28h] [rbp-68h]
  pthread_mutex_t *v50; // [rsp+30h] [rbp-60h]
  int v51; // [rsp+38h] [rbp-58h]
  __int16 v52; // [rsp+3Ch] [rbp-54h]
  char v53; // [rsp+3Fh] [rbp-51h]
  pthread_mutex_t *v54; // [rsp+40h] [rbp-50h]
  __int64 v55; // [rsp+48h] [rbp-48h]
  __int16 v56; // [rsp+50h] [rbp-40h] BYREF
  __int64 v57; // [rsp+58h] [rbp-38h]

  v5 = a1;
  v7 = a4;
  v51 = a5;
  v49 = a3;
  sub_1310140(a1, a2, (__int64 *)a3, a4, 0);
  v9 = (__int64)*a3;
  v10 = *((_WORD *)a3 + 10);
  v46 = &v44;
  v55 = v9;
  LOWORD(v54) = v10;
  LOWORD(v9) = v10 - v9;
  v52 = (unsigned __int16)v9 >> 3;
  v11 = ((unsigned __int16)v9 >> 3) - a5;
  v56 = v11;
  v57 = v55 + (unsigned __int16)v9 - 8LL * (unsigned __int16)v11;
  v12 = *(pthread_mutex_t **)(*a2 + 40);
  v50 = v12;
  v13 = alloca(16 * ((8 * (unsigned __int64)(v11 + 1) + 15) >> 4));
  v14 = &v44;
  sub_130FEB0((_QWORD *)v5, (__int64)&v56, v11, &v44);
  v15 = 16 * (3 * v7 - 108);
  v16 = v15 + 984;
  v17 = v15 + 968;
  if ( !v11 )
  {
    v48 = (volatile signed __int64 *)((char *)&v50->__list + v17);
    v47 = (volatile signed __int64 *)((char *)&v50->__list + v16);
LABEL_31:
    v43 = v49;
    _InterlockedAdd64(v48, (unsigned __int64)v49[1]);
    _InterlockedAdd64(v47, 1u);
    v43[1] = 0;
    goto LABEL_25;
  }
  v53 = 0;
  v47 = (volatile signed __int64 *)((char *)&v12->__list + v16);
  v48 = (volatile signed __int64 *)((char *)&v12->__list + v17);
  v45 = (unsigned __int64 *)(v5 + 112);
  while ( 1 )
  {
    v18 = **v14 & 0xFFFLL;
    v54 = (pthread_mutex_t *)qword_50579C0[*(_DWORD *)*v14 & 0xFFF];
    if ( v54[1973].__owner >= unk_5057900 )
    {
      v35 = (__int64)(&v54[263].__align + 2);
      if ( pthread_mutex_trylock(v54 + 265) )
      {
        sub_130AD90(v35);
        v54[266].__size[0] = 1;
      }
      v36 = v54;
      ++v54[264].__list.__next;
      if ( (struct __pthread_internal_list *)v5 != v36[264].__list.__prev )
      {
        ++*(&v36[264].__align + 2);
        v36[264].__list.__prev = (struct __pthread_internal_list *)v5;
      }
    }
    v19 = (v53 ^ 1) & (v50 == v54);
    if ( v19 )
    {
      v20 = v49;
      _InterlockedAdd64(v48, (unsigned __int64)v49[1]);
      _InterlockedAdd64(v47, 1u);
      v20[1] = 0;
      v53 = v19;
    }
    v21 = 0;
    do
    {
      while ( 1 )
      {
        v22 = v14[v21];
        if ( (_DWORD)v18 == (*(_DWORD *)v22 & 0xFFF) )
          break;
        if ( v11 <= (unsigned int)++v21 )
          goto LABEL_10;
      }
      ++v21;
      sub_130A0D0(v5, v22);
    }
    while ( v11 > (unsigned int)v21 );
LABEL_10:
    v23 = v54;
    if ( v54[1973].__owner >= unk_5057900 )
    {
      v54[266].__size[0] = 0;
      pthread_mutex_unlock(v23 + 265);
    }
    v24 = v14;
    LODWORD(v55) = 0;
    v25 = v5;
    v26 = 0;
    v27 = v24;
    do
    {
      while ( 1 )
      {
        v28 = v27[v26];
        if ( (_DWORD)v18 != (*(_DWORD *)v28 & 0xFFF) )
          break;
        ++v26;
        sub_130A0F0(v25, v28);
        if ( v11 <= (unsigned int)v26 )
          goto LABEL_16;
      }
      v29 = (unsigned int)v55;
      v30 = *(_QWORD *)(v57 + 8 * v26++);
      v31 = v55 + 1;
      *(_QWORD *)(v57 + 8LL * (unsigned int)v55) = v30;
      LODWORD(v55) = v31;
      v27[v29] = v28;
    }
    while ( v11 > (unsigned int)v26 );
LABEL_16:
    v32 = v27;
    v5 = v25;
    v33 = v11 - v55;
    v14 = v32;
    if ( v5 )
    {
      v34 = *(_DWORD *)(v5 + 152) - v33 < 0;
      *(_DWORD *)(v5 + 152) -= v33;
      if ( v34 )
      {
        if ( (unsigned __int8)sub_130FCA0((_DWORD *)(v5 + 152), v45) )
          sub_1315160(v5, v54, 0, 0);
      }
    }
    if ( !(_DWORD)v55 )
      break;
    v11 = v55;
  }
  if ( !v53 )
    goto LABEL_31;
LABEL_25:
  v37 = (__int64 *)v49;
  v38 = *v49;
  v39 = 8LL * (unsigned __int16)(v52 - v51);
  LOWORD(v55) = *((_WORD *)v49 + 10);
  memmove(
    &v38[v39],
    v38,
    8LL * (((unsigned __int16)(v55 - (_WORD)v38) >> 3) - (unsigned int)(unsigned __int16)(v52 - v51)));
  v40 = v39 + *v37;
  v41 = *((_WORD *)v37 + 10) - v40;
  result = (unsigned __int16)(*((_WORD *)v37 + 10) - *((_WORD *)v37 + 8)) >> 3;
  *v37 = v40;
  if ( (unsigned __int16)(v41 >> 3) < result )
    *((_WORD *)v37 + 8) = v40;
  return result;
}
