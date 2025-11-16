// Function: sub_16832F0
// Address: 0x16832f0
//
__int64 __fastcall sub_16832F0(__int64 *a1, __int64 a2)
{
  char *v3; // rsi
  __int64 v5; // rax
  __int64 v6; // r12
  const void *v7; // r14
  volatile unsigned __int32 v8; // r14d
  char *v10; // rcx
  signed __int64 v11; // rdx
  signed __int64 v12; // r15
  __int64 v13; // rax
  int v14; // edx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  char *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+18h] [rbp-48h] BYREF
  _QWORD *v20; // [rsp+20h] [rbp-40h] BYREF
  int pipedes[14]; // [rsp+28h] [rbp-38h] BYREF

  if ( !a1 )
    return 4;
  v3 = (char *)&unk_435FF63;
  v5 = sub_2207800(296, &unk_435FF63);
  v6 = v5;
  if ( !v5 )
    return 13;
  *(_BYTE *)(v5 + 8) = 1;
  *(_DWORD *)v5 = 0;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 88) = 0;
  *(_QWORD *)(v5 + 128) = 0;
  *(_OWORD *)(v5 + 56) = 0;
  *(_OWORD *)(v5 + 72) = 0;
  *(_OWORD *)(v5 + 96) = 0;
  *(_OWORD *)(v5 + 112) = 0;
  sub_2210B10(v5 + 136);
  *(_WORD *)(v6 + 184) = 0;
  *(_BYTE *)(v6 + 186) = 0;
  *(_QWORD *)(v6 + 188) = -1;
  *(_QWORD *)(v6 + 196) = -1;
  *(_WORD *)(v6 + 204) = 0;
  *(_QWORD *)(v6 + 208) = 0;
  *(_QWORD *)(v6 + 248) = 0;
  *(_QWORD *)(v6 + 288) = 0;
  *(_OWORD *)(v6 + 216) = 0;
  *(_OWORD *)(v6 + 232) = 0;
  *(_OWORD *)(v6 + 256) = 0;
  *(_OWORD *)(v6 + 272) = 0;
  sub_1682BF0(v6);
  if ( !*(_DWORD *)v6 )
  {
    *(_QWORD *)pipedes = -1;
    if ( pipe(pipedes) )
    {
      *(_DWORD *)(v6 + 4) = *__errno_location();
      _InterlockedCompareExchange((volatile signed __int32 *)v6, 11, 0);
    }
    else if ( pipedes[0] >= 0 && (v14 = pipedes[1], pipedes[1] >= 0) )
    {
      *(_DWORD *)(v6 + 196) = pipedes[0];
      *(_DWORD *)(v6 + 200) = v14;
      v19 = 0;
      v15 = (_QWORD *)sub_22077B0(16);
      if ( v15 )
      {
        v15[1] = v6;
        *v15 = &unk_49EE558;
      }
      v3 = (char *)&v20;
      v20 = v15;
      sub_22420C0(&v19, &v20, &pthread_create);
      v17 = v20;
      if ( v20 )
        (*(void (__fastcall **)(_QWORD *))(*v20 + 8LL))(v20);
      if ( *(_QWORD *)(v6 + 208) )
        sub_2207530(v17, &v20, v16);
      *(_QWORD *)(v6 + 208) = v19;
    }
    else
    {
      _InterlockedCompareExchange((volatile signed __int32 *)v6, 2, 0);
    }
  }
  if ( a2 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  v7 = *(const void **)(v6 + 32);
  if ( (unsigned __int64)a2 > *(_QWORD *)(v6 + 48) - (_QWORD)v7 )
  {
    v10 = 0;
    v11 = *(_QWORD *)(v6 + 40) - (_QWORD)v7;
    v12 = v11;
    if ( a2 )
    {
      v13 = sub_22077B0(a2);
      v7 = *(const void **)(v6 + 32);
      v10 = (char *)v13;
      v11 = *(_QWORD *)(v6 + 40) - (_QWORD)v7;
    }
    if ( v11 > 0 )
    {
      v10 = (char *)memmove(v10, v7, v11);
      v3 = (char *)(*(_QWORD *)(v6 + 48) - (_QWORD)v7);
    }
    else
    {
      if ( !v7 )
      {
LABEL_16:
        *(_QWORD *)(v6 + 32) = v10;
        *(_QWORD *)(v6 + 40) = &v10[v12];
        *(_QWORD *)(v6 + 48) = &v10[a2];
        goto LABEL_6;
      }
      v3 = (char *)(*(_QWORD *)(v6 + 48) - (_QWORD)v7);
    }
    v18 = v10;
    j_j___libc_free_0(v7, v3);
    v10 = v18;
    goto LABEL_16;
  }
LABEL_6:
  v8 = *(_DWORD *)v6;
  if ( *(_DWORD *)v6 )
  {
    sub_16823C0(v6, v3);
    j_j___libc_free_0(v6, 296);
  }
  else
  {
    *a1 = v6;
  }
  return v8;
}
