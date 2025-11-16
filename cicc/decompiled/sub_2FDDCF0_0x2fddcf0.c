// Function: sub_2FDDCF0
// Address: 0x2fddcf0
//
void __fastcall sub_2FDDCF0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r15
  __int64 v8; // r13
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // r14
  unsigned __int64 *v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned __int64 i; // rax
  __int64 v16; // rax
  _BYTE *v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+28h] [rbp-48h] BYREF
  _BYTE *v21; // [rsp+30h] [rbp-40h]
  __int64 v22; // [rsp+38h] [rbp-38h]
  _BYTE v23[48]; // [rsp+40h] [rbp-30h] BYREF

  v6 = a2;
  v8 = *(_QWORD *)(a2 + 24);
  if ( *(_DWORD *)(v8 + 120) )
  {
    do
    {
      sub_2E33590(v8, *(__int64 **)(v8 + 112), 0);
      a4 = *(unsigned int *)(v8 + 120);
    }
    while ( (_DWORD)a4 );
  }
  v9 = *(_QWORD *)(a2 + 56);
  v20 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v20, v9, 1);
  if ( a2 != v8 + 48 )
  {
    if ( (*(_BYTE *)v6 & 4) == 0 )
      goto LABEL_16;
    while ( 1 )
    {
      v19 = *(_BYTE **)(v6 + 8);
      if ( sub_2E88F60(v6) )
        goto LABEL_19;
      while ( 1 )
      {
        v10 = v6;
        if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
        {
          do
            v10 = *(_QWORD *)(v10 + 8);
          while ( (*(_BYTE *)(v10 + 44) & 8) != 0 );
        }
        v11 = *(_QWORD *)(v10 + 8);
        while ( v6 != v11 )
        {
          v12 = (_QWORD *)v6;
          v6 = *(_QWORD *)(v6 + 8);
          sub_2E31080(v8 + 40, (__int64)v12);
          v13 = (unsigned __int64 *)v12[1];
          v14 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
          *v13 = v14 | *v13 & 7;
          *(_QWORD *)(v14 + 8) = v13;
          *v12 &= 7uLL;
          v12[1] = 0;
          sub_2E310F0(v8 + 40);
        }
        a4 = (__int64)v19;
        if ( (_BYTE *)(v8 + 48) == v19 )
          goto LABEL_20;
        if ( !v19 )
          BUG();
        v6 = (unsigned __int64)v19;
        if ( (*v19 & 4) != 0 )
          break;
LABEL_16:
        for ( i = v6; (*(_BYTE *)(i + 44) & 8) != 0; i = *(_QWORD *)(i + 8) )
          ;
        v19 = *(_BYTE **)(i + 8);
        if ( sub_2E88F60(v6) )
LABEL_19:
          sub_2E79700(*(_QWORD *)(v8 + 32), v6);
      }
    }
  }
LABEL_20:
  if ( a3 != *(_QWORD *)(v8 + 8) )
  {
    v21 = v23;
    v16 = *a1;
    v22 = 0;
    (*(void (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(v16 + 368))(
      a1,
      v8,
      a3,
      0,
      v23,
      0,
      &v20,
      0);
    if ( v21 != v23 )
      _libc_free((unsigned __int64)v21);
  }
  sub_2E33F80(v8, a3, -1, a4, a5, a6);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
}
