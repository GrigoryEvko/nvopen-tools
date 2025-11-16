// Function: sub_35ECD10
// Address: 0x35ecd10
//
__int64 __fastcall sub_35ECD10(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (__fastcall *v5)(__int64, __int64, __int64, __int64, __int64); // rdx
  __int64 (*v7)(void); // rax
  __int64 (*v8)(); // rax
  __int64 v9; // r12
  int *v10; // rbx
  int v11; // eax
  int v12; // r14d
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *i; // rdx
  unsigned int v16; // r14d
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 (__fastcall *v21)(__int64, unsigned int, int); // rcx
  __int64 (__fastcall *v22)(__int64); // rax
  __int64 v23; // rax
  bool (__fastcall *v24)(__int64); // rdx
  unsigned int v26; // ecx
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *j; // rdx
  _QWORD *v35; // rax
  __int64 v36; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v37; // [rsp+Fh] [rbp-91h]
  char *v38; // [rsp+10h] [rbp-90h]
  __int64 v39; // [rsp+18h] [rbp-88h]
  int v40; // [rsp+18h] [rbp-88h]
  unsigned int v41; // [rsp+2Ch] [rbp-74h] BYREF
  char *v42; // [rsp+30h] [rbp-70h] BYREF
  int v43; // [rsp+38h] [rbp-68h]
  char v44; // [rsp+40h] [rbp-60h] BYREF

  v5 = sub_35ECCC0;
  v7 = (__int64 (*)(void))(*a1)[3];
  if ( (char *)v7 == (char *)sub_35ECCC0 )
  {
    v8 = *(__int64 (**)())(*a1[5] + 280LL);
    if ( v8 != sub_3059470 && !(unsigned __int8)v8() )
      return 0;
    v37 = sub_35EC3C0((__int64)a1, a2, (__int64)v5, a4, a5);
  }
  else
  {
    v37 = v7();
  }
  if ( !v37 )
    return 0;
  v36 = sub_C996C0("WindowSearch", 12, 0, 0);
  ((void (__fastcall *)(_QWORD **))(*a1)[4])(a1);
  v9 = ((__int64 (__fastcall *)(_QWORD **, _QWORD))(*a1)[2])(a1, 0);
  ((void (__fastcall *)(char **, _QWORD **, _QWORD, _QWORD))(*a1)[8])(
    &v42,
    a1,
    (unsigned int)dword_50407C8,
    (unsigned int)dword_50406E8);
  v10 = (int *)v42;
  v38 = &v42[4 * v43];
  if ( v42 != v38 )
  {
    while ( 1 )
    {
      v11 = *((_DWORD *)a1 + 64);
      v12 = *v10;
      a1[30] = (_QWORD *)((char *)a1[30] + 1);
      if ( !v11 )
      {
        if ( !*((_DWORD *)a1 + 65) )
          goto LABEL_12;
        v13 = *((unsigned int *)a1 + 66);
        if ( (unsigned int)v13 > 0x40 )
        {
          sub_C7D6A0((__int64)a1[31], 16LL * (unsigned int)v13, 8);
          a1[31] = 0;
          a1[32] = 0;
          *((_DWORD *)a1 + 66) = 0;
          goto LABEL_12;
        }
LABEL_9:
        v14 = a1[31];
        for ( i = &v14[2 * v13]; i != v14; v14 += 2 )
          *v14 = -4096;
        a1[32] = 0;
        goto LABEL_12;
      }
      v26 = 4 * v11;
      v13 = *((unsigned int *)a1 + 66);
      if ( (unsigned int)(4 * v11) < 0x40 )
        v26 = 64;
      if ( v26 >= (unsigned int)v13 )
        goto LABEL_9;
      v27 = v11 - 1;
      if ( !v27 )
        break;
      _BitScanReverse(&v27, v27);
      v28 = a1[31];
      v29 = (unsigned int)(1 << (33 - (v27 ^ 0x1F)));
      if ( (int)v29 < 64 )
        v29 = 64;
      if ( (_DWORD)v29 != (_DWORD)v13 )
        goto LABEL_40;
      a1[32] = 0;
      v35 = &v28[2 * v29];
      do
      {
        if ( v28 )
          *v28 = -4096;
        v28 += 2;
      }
      while ( v35 != v28 );
LABEL_12:
      v16 = *((_DWORD *)a1 + 1608) + v12;
      v17 = sub_35E71C0((__int64)a1, v16, *((_DWORD *)a1 + 1609));
      v39 = v18;
      (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v9 + 80LL))(v9, a1[3]);
      (*(void (__fastcall **)(__int64, _QWORD *, __int64, __int64, _QWORD))(*(_QWORD *)v9 + 96LL))(
        v9,
        a1[3],
        v17,
        v39,
        *((unsigned int *)a1 + 1609));
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 112LL))(v9);
      v41 = ((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD))(*a1)[11])(a1, v9, v16);
      if ( v41 != dword_5040368 )
      {
        ((void (__fastcall *)(_QWORD **, _QWORD, unsigned int *))(*a1)[12])(a1, v16, &v41);
        v19 = (__int64)*a1;
        v20 = v41;
        v21 = (__int64 (__fastcall *)(__int64, unsigned int, int))(*a1)[13];
        if ( v21 == sub_35EC370 )
        {
          if ( v16 == *((_DWORD *)a1 + 1608) )
          {
            *((_DWORD *)a1 + 1610) = v41;
            *((_DWORD *)a1 + 1611) = v16;
            *((_DWORD *)a1 + 1612) = v20;
          }
          else if ( v41 < *((_DWORD *)a1 + 1610) && v41 + dword_5040448 <= *((_DWORD *)a1 + 1612) )
          {
            sub_35EBE30((__int64)a1, v16, v41);
            v19 = (__int64)*a1;
          }
          goto LABEL_18;
        }
        v21((__int64)a1, v16, v41);
      }
      v19 = (__int64)*a1;
LABEL_18:
      ++v10;
      (*(void (__fastcall **)(_QWORD **))(v19 + 56))(a1);
      if ( v38 == (char *)v10 )
        goto LABEL_19;
    }
    v28 = a1[31];
    LODWORD(v29) = 64;
LABEL_40:
    v40 = v29;
    sub_C7D6A0((__int64)v28, 16LL * (unsigned int)v13, 8);
    v30 = ((((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 16;
    v31 = (v30
         | (((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 66) = v31;
    v32 = (_QWORD *)sub_C7D670(16 * v31, 8);
    v33 = *((unsigned int *)a1 + 66);
    a1[32] = 0;
    a1[31] = v32;
    for ( j = &v32[2 * v33]; j != v32; v32 += 2 )
    {
      if ( v32 )
        *v32 = -4096;
    }
    goto LABEL_12;
  }
LABEL_19:
  v22 = (__int64 (__fastcall *)(__int64))(*a1)[5];
  if ( v22 == sub_35E7110 )
  {
    (*(void (__fastcall **)(_QWORD *))(*a1[9] + 104LL))(a1[9]);
    (*(void (__fastcall **)(_QWORD *))(*a1[9] + 88LL))(a1[9]);
    sub_35E6FF0((__int64)a1);
  }
  else
  {
    v22((__int64)a1);
  }
  v23 = (__int64)*a1;
  v24 = (bool (__fastcall *)(__int64))(*a1)[14];
  if ( v24 != sub_35E4D20 )
  {
    if ( v24((__int64)a1) )
    {
      v23 = (__int64)*a1;
      goto LABEL_23;
    }
LABEL_50:
    v37 = 0;
    goto LABEL_24;
  }
  if ( *((_DWORD *)a1 + 1611) == *((_DWORD *)a1 + 1608) )
    goto LABEL_50;
LABEL_23:
  (*(void (__fastcall **)(_QWORD **))(v23 + 120))(a1);
LABEL_24:
  if ( v42 != &v44 )
    _libc_free((unsigned __int64)v42);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  if ( v36 )
    sub_C9AF60(v36);
  return v37;
}
