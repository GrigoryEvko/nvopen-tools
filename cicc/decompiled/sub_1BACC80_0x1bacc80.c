// Function: sub_1BACC80
// Address: 0x1bacc80
//
void __fastcall sub_1BACC80(__int64 a1, unsigned int a2, int a3)
{
  int v4; // ebx
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 *v7; // r13
  __int64 *i; // r15
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  _QWORD *v11; // rdx
  __int64 v12; // rbx
  unsigned int v14; // [rsp+1Ch] [rbp-C4h]
  __int64 v15; // [rsp+20h] [rbp-C0h] BYREF
  int v16; // [rsp+28h] [rbp-B8h] BYREF
  unsigned int v17; // [rsp+2Ch] [rbp-B4h]
  __int64 v18; // [rsp+30h] [rbp-B0h] BYREF
  char *v19; // [rsp+38h] [rbp-A8h]
  char *v20; // [rsp+40h] [rbp-A0h]
  __int64 v21; // [rsp+48h] [rbp-98h]
  int v22; // [rsp+50h] [rbp-90h]
  char v23; // [rsp+58h] [rbp-88h] BYREF
  __int64 v24; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v25; // [rsp+68h] [rbp-78h]
  _BYTE *v26; // [rsp+70h] [rbp-70h]
  __int64 v27; // [rsp+78h] [rbp-68h]
  int v28; // [rsp+80h] [rbp-60h]
  _BYTE v29[88]; // [rsp+88h] [rbp-58h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)a1;
  v18 = 0;
  v19 = &v23;
  v20 = &v23;
  v21 = 1;
  v22 = 0;
  v6 = sub_13FCB50(v5);
  v7 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  for ( i = *(__int64 **)(*(_QWORD *)a1 + 32LL); v7 != i; ++i )
  {
    if ( v6 != *i )
    {
      v9 = sub_157EBA0(*i);
      if ( *(_BYTE *)(v9 + 16) == 26 && (*(_DWORD *)(v9 + 20) & 0xFFFFFFF) == 3 )
        sub_1412190((__int64)&v18, *(_QWORD *)(v9 - 72));
    }
  }
  v24 = 0;
  v25 = v29;
  v26 = v29;
  v27 = 4;
  v28 = 0;
  sub_1B96260((__int64 *)a1, (__int64)&v24);
  v14 = a3 + 1;
  if ( a2 < v14 )
  {
    do
    {
      while ( 1 )
      {
        v16 = v4;
        v17 = v14;
        sub_1BAB460(&v15, (__int64 *)a1, &v16, (__int64)&v18, (__int64)&v24);
        v10 = *(_DWORD *)(a1 + 56);
        if ( v10 >= *(_DWORD *)(a1 + 60) )
        {
          sub_1B98CA0(a1 + 48, 0);
          v10 = *(_DWORD *)(a1 + 56);
        }
        v11 = (_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * v10);
        if ( !v11 )
          break;
        *v11 = v15;
        ++*(_DWORD *)(a1 + 56);
LABEL_10:
        v4 = v17;
        if ( v17 >= v14 )
          goto LABEL_16;
      }
      v12 = v15;
      *(_DWORD *)(a1 + 56) = v10 + 1;
      if ( !v12 )
        goto LABEL_10;
      sub_1B949D0(v12);
      j_j___libc_free_0(v12, 472);
      v4 = v17;
    }
    while ( v17 < v14 );
  }
LABEL_16:
  if ( v26 != v25 )
    _libc_free((unsigned __int64)v26);
  if ( v20 != v19 )
    _libc_free((unsigned __int64)v20);
}
