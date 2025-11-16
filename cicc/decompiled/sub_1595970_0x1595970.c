// Function: sub_1595970
// Address: 0x1595970
//
void __fastcall sub_1595970(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rdx
  int v4; // eax
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx

  v1 = ***(_QWORD ***)a1;
  v2 = sub_1595920(a1);
  v4 = sub_16D1B30(v1 + 1712, v2, v3);
  if ( v4 == -1 )
    v5 = *(_QWORD *)(v1 + 1712) + 8LL * *(unsigned int *)(v1 + 1720);
  else
    v5 = *(_QWORD *)(v1 + 1712) + 8LL * v4;
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 8LL);
  v7 = *(_QWORD *)(v6 + 32);
  if ( v7 )
  {
    if ( v6 == a1 )
    {
      v11 = (_QWORD *)(*(_QWORD *)v5 + 8LL);
    }
    else
    {
      while ( 1 )
      {
        v8 = v7;
        v7 = *(_QWORD *)(v7 + 32);
        if ( a1 == v8 )
          break;
        v6 = v8;
      }
      v11 = (_QWORD *)(v6 + 32);
    }
    *v11 = v7;
    *(_QWORD *)(a1 + 32) = 0;
  }
  else
  {
    v9 = (_QWORD *)sub_16498A0(a1);
    v10 = *(_QWORD *)v5;
    sub_16D1CB0(*v9 + 1712LL, v10);
    _libc_free(v10);
    *(_QWORD *)(a1 + 32) = 0;
  }
}
