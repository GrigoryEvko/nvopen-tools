// Function: sub_A09940
// Address: 0xa09940
//
__int64 *__fastcall sub_A09940(__int64 *a1, __int64 a2, __int64 **a3)
{
  unsigned __int64 v4; // rax
  __int64 *v5; // r8
  unsigned __int64 v6; // rax
  _BYTE *v8; // rcx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  unsigned __int64 v14; // rsi
  int v15; // r15d
  __int64 v16; // r9
  unsigned int v17; // r8d
  _DWORD *v18; // rdi
  int v19; // edx
  int v21; // ecx
  int v22; // ecx
  __int64 v23; // r10
  int v24; // edx
  _DWORD *v25; // rax
  int v26; // edi
  int v27; // edi
  int v28; // esi
  _DWORD *v29; // r8
  __int64 v30; // r9
  int v31; // r10d
  __int64 v32; // rcx
  int v33; // edi
  int v34; // r9d
  __int64 *v35; // [rsp+10h] [rbp-90h]
  int v36; // [rsp+10h] [rbp-90h]
  int v37; // [rsp+10h] [rbp-90h]
  _BYTE *v38; // [rsp+20h] [rbp-80h] BYREF
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+30h] [rbp-70h]
  _BYTE v41[8]; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v42[4]; // [rsp+40h] [rbp-60h] BYREF
  char v43; // [rsp+60h] [rbp-40h]
  char v44; // [rsp+61h] [rbp-3Fh]

  v4 = *((unsigned int *)a3 + 2);
  if ( v4 <= 1 )
  {
    v44 = 1;
    v42[0] = "Invalid record";
    v43 = 3;
    sub_A01DB0(a1, (__int64)v42);
    return a1;
  }
  v5 = *a3;
  v6 = 8 * v4;
  v8 = v41;
  v9 = **a3;
  v10 = (__int64)(v6 - 8) >> 3;
  v38 = v41;
  v39 = 0;
  v40 = 8;
  if ( v6 > 0x48 )
  {
    v35 = v5;
    sub_C8D290(&v38, v41, (__int64)(v6 - 8) >> 3, 1);
    v5 = v35;
    v8 = &v38[v39];
  }
  v11 = 0;
  do
  {
    v8[v11] = v5[v11 + 1];
    ++v11;
  }
  while ( v10 != v11 );
  v12 = *(_QWORD *)(a2 + 256);
  v39 += v10;
  v13 = sub_BA8BE0(v12, v38);
  v14 = *(unsigned int *)(a2 + 1088);
  v15 = v13;
  if ( !(_DWORD)v14 )
  {
    ++*(_QWORD *)(a2 + 1064);
    goto LABEL_14;
  }
  v16 = *(_QWORD *)(a2 + 1072);
  v17 = (v14 - 1) & (37 * v9);
  v18 = (_DWORD *)(v16 + 8LL * v17);
  v19 = *v18;
  if ( (_DWORD)v9 != *v18 )
  {
    v36 = 1;
    v25 = 0;
    while ( v19 != -1 )
    {
      if ( v25 || v19 != -2 )
        v18 = v25;
      v17 = (v14 - 1) & (v36 + v17);
      v19 = *(_DWORD *)(v16 + 8LL * v17);
      if ( (_DWORD)v9 == v19 )
        goto LABEL_8;
      ++v36;
      v25 = v18;
      v18 = (_DWORD *)(v16 + 8LL * v17);
    }
    if ( !v25 )
      v25 = v18;
    v27 = *(_DWORD *)(a2 + 1080);
    ++*(_QWORD *)(a2 + 1064);
    v24 = v27 + 1;
    if ( 4 * (v27 + 1) < (unsigned int)(3 * v14) )
    {
      if ( (int)v14 - *(_DWORD *)(a2 + 1084) - v24 > (unsigned int)v14 >> 3 )
        goto LABEL_16;
      v37 = 37 * v9;
      sub_A09770(a2 + 1064, v14);
      v28 = *(_DWORD *)(a2 + 1088);
      if ( v28 )
      {
        v14 = (unsigned int)(v28 - 1);
        v29 = 0;
        v30 = *(_QWORD *)(a2 + 1072);
        v31 = 1;
        LODWORD(v32) = v14 & v37;
        v24 = *(_DWORD *)(a2 + 1080) + 1;
        v25 = (_DWORD *)(v30 + 8LL * ((unsigned int)v14 & v37));
        v33 = *v25;
        if ( (_DWORD)v9 != *v25 )
        {
          while ( v33 != -1 )
          {
            if ( !v29 && v33 == -2 )
              v29 = v25;
            v32 = (unsigned int)v14 & ((_DWORD)v32 + v31);
            v25 = (_DWORD *)(v30 + 8 * v32);
            v33 = *v25;
            if ( (_DWORD)v9 == *v25 )
              goto LABEL_16;
            ++v31;
          }
LABEL_28:
          if ( v29 )
            v25 = v29;
          goto LABEL_16;
        }
        goto LABEL_16;
      }
      goto LABEL_48;
    }
LABEL_14:
    sub_A09770(a2 + 1064, 2 * v14);
    v21 = *(_DWORD *)(a2 + 1088);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 1072);
      v14 = v22 & (unsigned int)(37 * v9);
      v24 = *(_DWORD *)(a2 + 1080) + 1;
      v25 = (_DWORD *)(v23 + 8 * v14);
      v26 = *v25;
      if ( (_DWORD)v9 != *v25 )
      {
        v34 = 1;
        v29 = 0;
        while ( v26 != -1 )
        {
          if ( v26 == -2 && !v29 )
            v29 = v25;
          v14 = v22 & (unsigned int)(v14 + v34);
          v25 = (_DWORD *)(v23 + 8 * v14);
          v26 = *v25;
          if ( (_DWORD)v9 == *v25 )
            goto LABEL_16;
          ++v34;
        }
        goto LABEL_28;
      }
LABEL_16:
      *(_DWORD *)(a2 + 1080) = v24;
      if ( *v25 != -1 )
        --*(_DWORD *)(a2 + 1084);
      *v25 = v9;
      v25[1] = v15;
      *a1 = 1;
      goto LABEL_9;
    }
LABEL_48:
    ++*(_DWORD *)(a2 + 1080);
    BUG();
  }
LABEL_8:
  v14 = (unsigned __int64)v42;
  v44 = 1;
  v42[0] = "Conflicting METADATA_KIND records";
  v43 = 3;
  sub_A01DB0(a1, (__int64)v42);
LABEL_9:
  if ( v38 != v41 )
    _libc_free(v38, v14);
  return a1;
}
