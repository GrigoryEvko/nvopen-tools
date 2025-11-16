// Function: sub_37BCE70
// Address: 0x37bce70
//
void __fastcall sub_37BCE70(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  int v8; // eax
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rdx
  char *v17; // rax
  int v18; // eax
  _BYTE v19[48]; // [rsp+0h] [rbp-170h] BYREF
  __int64 v20; // [rsp+30h] [rbp-140h] BYREF
  char *v21; // [rsp+38h] [rbp-138h]
  __int64 v22; // [rsp+40h] [rbp-130h]
  int v23; // [rsp+48h] [rbp-128h]
  char v24; // [rsp+4Ch] [rbp-124h]
  char v25; // [rsp+50h] [rbp-120h] BYREF

  v7 = *a1;
  v21 = &v25;
  v20 = 0;
  v8 = *(_DWORD *)(v7 + 608);
  v24 = 1;
  v22 = 32;
  v23 = 0;
  if ( !v8 )
    goto LABEL_14;
  v10 = 0;
  v11 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(*(_QWORD *)(v7 + 600) + 8 * v11);
        v13 = *(_QWORD *)a1[1] + 80LL * *(int *)(v12 + 24);
        if ( (*(_BYTE *)(v13 + 8) & 1) != 0 )
        {
          v14 = v13 + 16;
          v15 = 3;
          goto LABEL_5;
        }
        v14 = *(_QWORD *)(v13 + 16);
        v18 = *(_DWORD *)(v13 + 24);
        if ( v18 )
          break;
LABEL_11:
        v11 = (unsigned int)(v10 + 1);
        v10 = v11;
        if ( *(_DWORD *)(v7 + 608) <= (unsigned int)v11 )
          goto LABEL_12;
      }
      v15 = v18 - 1;
LABEL_5:
      a5 = v15 & a2;
      v16 = *(unsigned int *)(v14 + 16 * a5);
      if ( a2 != (_DWORD)v16 )
      {
        a6 = 1;
        while ( (_DWORD)v16 != -1 )
        {
          a5 = v15 & (unsigned int)(a6 + a5);
          v16 = *(unsigned int *)(v14 + 16LL * (unsigned int)a5);
          if ( a2 == (_DWORD)v16 )
            goto LABEL_6;
          a6 = (unsigned int)(a6 + 1);
        }
        goto LABEL_11;
      }
LABEL_6:
      if ( v24 )
        break;
LABEL_18:
      sub_C8CC70((__int64)&v20, v12, v16, v7, a5, a6);
      v7 = *a1;
      v11 = (unsigned int)(v10 + 1);
      v10 = v11;
      if ( *(_DWORD *)(*a1 + 608) <= (unsigned int)v11 )
        goto LABEL_12;
    }
    v17 = v21;
    v16 = (__int64)&v21[8 * HIDWORD(v22)];
    if ( v21 != (char *)v16 )
    {
      while ( v12 != *(_QWORD *)v17 )
      {
        v17 += 8;
        if ( (char *)v16 == v17 )
          goto LABEL_20;
      }
      goto LABEL_11;
    }
LABEL_20:
    if ( HIDWORD(v22) >= (unsigned int)v22 )
      goto LABEL_18;
    v11 = (unsigned int)(v10 + 1);
    ++HIDWORD(v22);
    v10 = v11;
    *(_QWORD *)v16 = v12;
    v7 = *a1;
    ++v20;
  }
  while ( *(_DWORD *)(v7 + 608) > (unsigned int)v11 );
LABEL_12:
  if ( HIDWORD(v22) != v23 )
    sub_37BC2F0((__int64)v19, (__int64)&v20, *(__int64 **)(a1[2] + 328), v7, a5, a6);
LABEL_14:
  *(_DWORD *)(a1[3] + 8) = 0;
  sub_37BCE40(*a1, a1[4], (__int64)&v20, a1[3]);
  if ( !v24 )
    _libc_free((unsigned __int64)v21);
}
