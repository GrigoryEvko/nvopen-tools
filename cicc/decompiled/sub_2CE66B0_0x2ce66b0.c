// Function: sub_2CE66B0
// Address: 0x2ce66b0
//
void __fastcall sub_2CE66B0(_QWORD *a1, unsigned __int64 a2, int a3, _QWORD *a4)
{
  _QWORD *v5; // r9
  _QWORD *v8; // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  int *v13; // rax
  int *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  int *v17; // rax
  unsigned __int64 v18; // [rsp+8h] [rbp-28h] BYREF
  unsigned __int64 *v19; // [rsp+18h] [rbp-18h] BYREF

  v5 = a1 + 1;
  v8 = (_QWORD *)a1[2];
  v18 = a2;
  if ( !v8 )
  {
    v10 = (__int64)(a1 + 1);
LABEL_17:
    v19 = &v18;
    v10 = sub_2CE6580(a1, v10, &v19);
    goto LABEL_8;
  }
  v10 = (__int64)v5;
  do
  {
    while ( 1 )
    {
      v11 = v8[2];
      v12 = v8[3];
      if ( v8[4] >= a2 )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v12 )
        goto LABEL_6;
    }
    v10 = (__int64)v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v11 );
LABEL_6:
  if ( v5 == (_QWORD *)v10 || *(_QWORD *)(v10 + 32) > a2 )
    goto LABEL_17;
LABEL_8:
  *(_DWORD *)(v10 + 40) = a3;
  v13 = (int *)a4[2];
  if ( v13 )
  {
    v14 = (int *)(a4 + 1);
    do
    {
      while ( 1 )
      {
        v15 = *((_QWORD *)v13 + 2);
        v16 = *((_QWORD *)v13 + 3);
        if ( *((_QWORD *)v13 + 4) >= v18 )
          break;
        v13 = (int *)*((_QWORD *)v13 + 3);
        if ( !v16 )
          goto LABEL_13;
      }
      v14 = v13;
      v13 = (int *)*((_QWORD *)v13 + 2);
    }
    while ( v15 );
LABEL_13:
    if ( a4 + 1 != (_QWORD *)v14 && *((_QWORD *)v14 + 4) <= v18 )
    {
      v17 = sub_220F330(v14, a4 + 1);
      j_j___libc_free_0((unsigned __int64)v17);
      --a4[5];
    }
  }
}
