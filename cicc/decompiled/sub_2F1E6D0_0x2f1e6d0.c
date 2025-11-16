// Function: sub_2F1E6D0
// Address: 0x2f1e6d0
//
void __fastcall sub_2F1E6D0(__int64 a1, unsigned __int64 a2)
{
  _DWORD *v2; // rax
  const void *v3; // r15
  __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // r8
  unsigned __int64 v10; // r9
  char *v11; // rcx
  char *v12; // rax
  _DWORD *v13; // rdx
  char *v14; // rax
  unsigned __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp-48h] [rbp-48h]
  unsigned __int64 v18; // [rsp-40h] [rbp-40h]
  char *v19; // [rsp-40h] [rbp-40h]
  unsigned __int64 v20; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v2 = *(_DWORD **)(a1 + 8);
    v3 = *(const void **)a1;
    v4 = (__int64)v2 - *(_QWORD *)a1;
    v5 = 0xCCCCCCCCCCCCCCCDLL * (v4 >> 2);
    if ( a2 > 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v2) >> 2) )
    {
      if ( 0x666666666666666LL - v5 < a2 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v6 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v2 - *(_QWORD *)a1) >> 2);
      if ( a2 >= v5 )
        v6 = a2;
      v7 = __CFADD__(v5, v6);
      v8 = v5 + v6;
      if ( v7 )
      {
        v15 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v8 )
        {
          v9 = v4;
          v10 = 0;
          v11 = 0;
LABEL_9:
          v12 = &v11[v4];
          do
          {
            *(_DWORD *)v12 = 0;
            v12 += 20;
            *((_DWORD *)v12 - 4) = 0;
            *((_DWORD *)v12 - 3) = 0;
            *((_DWORD *)v12 - 2) = 0;
            *((_DWORD *)v12 - 1) = 0;
          }
          while ( &v11[20 * a2 + v4] != v12 );
          if ( v9 > 0 )
          {
            v18 = v10;
            v14 = (char *)memmove(v11, v3, v9);
            v10 = v18;
            v11 = v14;
          }
          else if ( !v3 )
          {
LABEL_13:
            *(_QWORD *)a1 = v11;
            *(_QWORD *)(a1 + 16) = v10;
            *(_QWORD *)(a1 + 8) = &v11[20 * v5 + 20 * a2];
            return;
          }
          v17 = v10;
          v19 = v11;
          j_j___libc_free_0((unsigned __int64)v3);
          v10 = v17;
          v11 = v19;
          goto LABEL_13;
        }
        if ( v8 > 0x666666666666666LL )
          v8 = 0x666666666666666LL;
        v15 = 20 * v8;
      }
      v20 = v15;
      v16 = sub_22077B0(v15);
      v3 = *(const void **)a1;
      v11 = (char *)v16;
      v10 = v16 + v20;
      v9 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
      goto LABEL_9;
    }
    v13 = &v2[5 * a2];
    do
    {
      *v2 = 0;
      v2 += 5;
      *(v2 - 4) = 0;
      *(v2 - 3) = 0;
      *(v2 - 2) = 0;
      *(v2 - 1) = 0;
    }
    while ( v13 != v2 );
    *(_QWORD *)(a1 + 8) = v13;
  }
}
