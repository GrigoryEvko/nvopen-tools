// Function: sub_1345880
// Address: 0x1345880
//
unsigned __int64 *__fastcall sub_1345880(
        _BYTE *a1,
        __int64 *a2,
        unsigned int *a3,
        __int64 a4,
        __int64 *a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        char a8,
        char *a9,
        int a10,
        char a11)
{
  pthread_mutex_t *v11; // r15
  int v14; // eax
  __int64 *v15; // rdx
  __int64 v16; // r9
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rbx
  __int64 v19; // r9
  unsigned __int64 *v20; // rbx
  __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  __int64 v23; // rdi
  unsigned __int64 *v24; // rbx
  int v26; // r8d
  unsigned __int64 *v27; // rax
  __int64 *v28; // r8
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v34; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 *v35; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 *v36; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 *v37; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 *v38; // [rsp+48h] [rbp-38h] BYREF

  v11 = (pthread_mutex_t *)(a4 + 64);
  v14 = pthread_mutex_trylock((pthread_mutex_t *)(a4 + 64));
  v15 = a5;
  if ( v14 )
  {
    sub_130AD90(a4);
    *(_BYTE *)(a4 + 104) = 1;
    v15 = a5;
  }
  ++*(_QWORD *)(a4 + 56);
  if ( a1 != *(_BYTE **)(a4 + 48) )
  {
    ++*(_QWORD *)(a4 + 40);
    *(_QWORD *)(a4 + 48) = a1;
  }
  v16 = a4 + 112;
  if ( a11 )
    v16 = a4 + 9768;
  if ( v15 )
  {
    v30 = v16;
    v17 = (unsigned __int64 *)sub_1341B70((__int64)a1, a2[7298], v15, 0, *(_DWORD *)(a4 + 19424));
    v18 = v17;
    if ( v17 )
    {
      v19 = v30;
      if ( a6 <= (v17[2] & 0xFFFFFFFFFFFFF000LL) )
        goto LABEL_10;
      sub_1341B90((__int64)a1, a2[7298], v17, *(_DWORD *)(a4 + 19424));
    }
LABEL_31:
    *(_BYTE *)(a4 + 104) = 0;
    v24 = 0;
    pthread_mutex_unlock(v11);
    return v24;
  }
  v26 = 64;
  if ( *(_BYTE *)(a4 + 19432) )
    v26 = qword_4C6F288;
  v31 = v16;
  v27 = sub_1342C60(v16, a6, a7, a11, v26);
  v19 = v31;
  v18 = v27;
  if ( !v27 )
    goto LABEL_31;
LABEL_10:
  sub_1342A40(v19, v18);
  sub_1341570((__int64)a1, a2[7298], v18, 0);
  v34 = v18;
  v37 = 0;
  v38 = 0;
  if ( (unsigned int)sub_1343C70(a1, (__int64)a2, a3, &v34, &v35, &v36, &v37, &v38, a6, a7) )
  {
    if ( v38 )
      sub_1341E90((__int64)a1, a2[7298], (__int64)v38);
    v24 = v37;
    if ( !v37 )
    {
      *(_BYTE *)(a4 + 104) = 0;
      pthread_mutex_unlock(v11);
      return v24;
    }
    sub_1341E90((__int64)a1, a2[7298], (__int64)v37);
    *(_BYTE *)(a4 + 104) = 0;
    pthread_mutex_unlock(v11);
    sub_1343DD0(a1, (__int64)a2, a3, a4, v37);
    if ( pthread_mutex_trylock(v11) )
    {
      sub_130AD90(a4);
      *(_BYTE *)(a4 + 104) = 1;
    }
    ++*(_QWORD *)(a4 + 56);
    if ( a1 != *(_BYTE **)(a4 + 48) )
    {
      ++*(_QWORD *)(a4 + 40);
      *(_QWORD *)(a4 + 48) = a1;
    }
    goto LABEL_31;
  }
  v20 = v35;
  if ( v35 )
  {
    sub_1341570((__int64)a1, a2[7298], v35, *(_DWORD *)(a4 + 19424));
    v21 = a4 + 112;
    if ( (*v20 & 0x10000) != 0 )
      v21 = a4 + 9768;
    sub_1342830(v21, v20);
  }
  v22 = v36;
  if ( v36 )
  {
    sub_1341570((__int64)a1, a2[7298], v36, *(_DWORD *)(a4 + 19424));
    v23 = a4 + 112;
    if ( (*v22 & 0x10000) != 0 )
      v23 = a4 + 9768;
    sub_1342830(v23, v22);
  }
  v24 = v34;
  *(_BYTE *)(a4 + 104) = 0;
  pthread_mutex_unlock(v11);
  if ( v24 )
  {
    if ( (unsigned __int8)sub_13457C0(a1, (__int64)a3, (__int64 *)v24, *a9, a8) )
    {
      v28 = (__int64 *)v24;
      v24 = 0;
      sub_13451C0(a1, a2, a3, a4, v28);
    }
    else if ( (*v24 & 0x2000) != 0 )
    {
      *a9 = 1;
    }
  }
  return v24;
}
