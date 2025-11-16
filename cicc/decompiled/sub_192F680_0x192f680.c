// Function: sub_192F680
// Address: 0x192f680
//
void __fastcall sub_192F680(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned __int64 v11; // r12
  __int64 v12; // r14
  size_t v13; // r15
  unsigned __int64 v14; // r9
  int v15; // r8d
  int v16; // eax
  const void *v17; // rdi
  size_t v18; // r11
  int v19; // eax
  __int64 n; // [rsp+18h] [rbp-158h]
  void *v21; // [rsp+20h] [rbp-150h]
  void *v22; // [rsp+28h] [rbp-148h]
  _BYTE *v23; // [rsp+38h] [rbp-138h]
  __int64 v24; // [rsp+40h] [rbp-130h]
  void *s2; // [rsp+48h] [rbp-128h]
  __int64 v26; // [rsp+50h] [rbp-120h]
  unsigned __int64 v27; // [rsp+58h] [rbp-118h]
  void *s1a; // [rsp+60h] [rbp-110h]
  void *s1; // [rsp+60h] [rbp-110h]
  __int64 v30; // [rsp+68h] [rbp-108h]
  int v31; // [rsp+70h] [rbp-100h]
  int v32; // [rsp+74h] [rbp-FCh]
  int v33; // [rsp+78h] [rbp-F8h]
  _BYTE *v34; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+88h] [rbp-E8h]
  _BYTE v36[32]; // [rsp+90h] [rbp-E0h] BYREF
  _BYTE *v37; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+B8h] [rbp-B8h]
  _BYTE v39[32]; // [rsp+C0h] [rbp-B0h] BYREF
  _BYTE *v40; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v41; // [rsp+E8h] [rbp-88h]
  _BYTE v42[32]; // [rsp+F0h] [rbp-80h] BYREF
  _BYTE *v43; // [rsp+110h] [rbp-60h] BYREF
  __int64 v44; // [rsp+118h] [rbp-58h]
  _BYTE v45[80]; // [rsp+120h] [rbp-50h] BYREF

  if ( !byte_4FAF3A0 && (unsigned int)sub_2207590(&byte_4FAF3A0) )
  {
    qword_4FAF3D0 = 0;
    qword_4FAF3C0 = (__int64)&qword_4FAF3D0;
    qword_4FAF3F0 = (__int64)&unk_4FAF400;
    qword_4FAF3F8 = 0x400000000LL;
    qword_4FAF3C8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF3C0, &qword_4A427C0);
    sub_2207640(&byte_4FAF3A0);
  }
  v34 = v36;
  v35 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3C8 )
    sub_192DAF0((__int64)&v34, (__int64)&qword_4FAF3C0, a3, a4, a5, a6);
  v7 = (unsigned int)qword_4FAF3F8;
  v37 = v39;
  v38 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3F8 )
    sub_192DA10((__int64)&v37, (__int64)&qword_4FAF3F0, a3, (unsigned int)qword_4FAF3F8, a5, a6);
  if ( !byte_4FAF328 && (unsigned int)sub_2207590(&byte_4FAF328) )
  {
    qword_4FAF350 = 1;
    qword_4FAF340 = (__int64)&qword_4FAF350;
    qword_4FAF370 = (__int64)&unk_4FAF380;
    qword_4FAF378 = 0x400000000LL;
    qword_4FAF348 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF340, &qword_4A427C0);
    sub_2207640(&byte_4FAF328);
  }
  v8 = (unsigned int)qword_4FAF348;
  v40 = v42;
  v41 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF348 )
    sub_192DAF0((__int64)&v40, (__int64)&qword_4FAF340, (unsigned int)qword_4FAF348, v7, a5, a6);
  v43 = v45;
  v44 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF378 )
  {
    sub_192DA10((__int64)&v43, (__int64)&qword_4FAF370, v8, v7, a5, a6);
    v9 = *(_QWORD *)(a1 + 16);
    v10 = *(_QWORD *)(a1 + 24);
    if ( v10 != v9 )
    {
      v26 = (unsigned int)v44;
      v23 = v43;
      goto LABEL_11;
    }
    v23 = v43;
LABEL_14:
    if ( v23 != v45 )
      _libc_free((unsigned __int64)v23);
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 16);
    v26 = 0;
    v10 = *(_QWORD *)(a1 + 24);
    v23 = v45;
    if ( v10 != v9 )
    {
LABEL_11:
      v11 = (unsigned int)v35;
      s2 = v34;
      v21 = v37;
      v32 = v41;
      v22 = v40;
      n = 8 * v26;
      v30 = (unsigned int)v38;
      v24 = 8LL * (unsigned int)v38;
      v12 = v9;
      v13 = 8LL * (unsigned int)v35;
      while ( 1 )
      {
        v14 = *(unsigned int *)(v12 + 8);
        v15 = *(_DWORD *)(v12 + 8);
        if ( v14 != v11 )
          break;
        if ( !v13 )
          goto LABEL_48;
        v27 = *(unsigned int *)(v12 + 8);
        v31 = *(_DWORD *)(v12 + 8);
        s1a = *(void **)v12;
        v16 = memcmp(*(const void **)v12, s2, v13);
        v17 = s1a;
        v15 = v31;
        v18 = v13;
        v14 = v27;
        if ( v16 )
        {
          if ( v32 != v31 )
            goto LABEL_14;
LABEL_26:
          if ( memcmp(v17, v22, v18) )
            goto LABEL_14;
LABEL_27:
          if ( *(_DWORD *)(v12 + 56) != v26 || n && memcmp(*(const void **)(v12 + 48), v23, n) )
            goto LABEL_14;
LABEL_30:
          v12 += 96;
          *(_QWORD *)(a1 + 16) = v12;
          if ( v12 == v10 )
            goto LABEL_14;
        }
        else
        {
LABEL_48:
          if ( *(_DWORD *)(v12 + 56) != v30 )
            break;
          s1 = (void *)v14;
          v33 = v15;
          if ( !v24 )
            goto LABEL_30;
          v19 = memcmp(*(const void **)(v12 + 48), v21, v24);
          v15 = v33;
          v14 = (unsigned __int64)s1;
          if ( v19 )
            break;
          v12 += 96;
          *(_QWORD *)(a1 + 16) = v12;
          if ( v12 == v10 )
            goto LABEL_14;
        }
      }
      if ( v32 != v15 )
        goto LABEL_14;
      v17 = *(const void **)v12;
      v18 = 8 * v14;
      if ( !(8 * v14) )
        goto LABEL_27;
      goto LABEL_26;
    }
  }
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
