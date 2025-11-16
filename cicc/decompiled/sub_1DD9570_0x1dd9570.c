// Function: sub_1DD9570
// Address: 0x1dd9570
//
void __fastcall sub_1DD9570(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rcx
  __int64 *v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rsi
  _DWORD *v10; // rbx
  _DWORD *v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // eax

  if ( a2 == a3 )
    return;
  v6 = *(__int64 **)(a1 + 96);
  v7 = *(__int64 **)(a1 + 88);
  if ( v6 == v7 )
  {
LABEL_14:
    sub_1DD9100(a2, a1);
    sub_1DD8D00(a3, (char *)a1);
    *v7 = a3;
    return;
  }
  v8 = *(__int64 **)(a1 + 88);
  v9 = *(_QWORD *)(a1 + 96);
  v7 = (__int64 *)v9;
  do
  {
    while ( 1 )
    {
      if ( *v8 == a2 )
      {
        v7 = v8;
        if ( v6 != (__int64 *)v9 )
          goto LABEL_8;
        goto LABEL_5;
      }
      if ( *v8 == a3 )
        break;
LABEL_5:
      if ( v6 == ++v8 )
        goto LABEL_13;
    }
    v9 = (__int64)v8;
    if ( v6 != v7 )
      goto LABEL_8;
    ++v8;
  }
  while ( v6 != v8 );
LABEL_13:
  if ( (__int64 *)v9 == v6 )
    goto LABEL_14;
LABEL_8:
  if ( *(_QWORD *)(a1 + 120) != *(_QWORD *)(a1 + 112) )
  {
    v10 = (_DWORD *)sub_1DD7680(a1, v9);
    if ( *v10 != -1 )
    {
      v11 = (_DWORD *)sub_1DD7680(a1, (__int64)v7);
      v12 = (unsigned int)*v11 + (unsigned __int64)(unsigned int)*v10;
      v13 = *v10 + *v11;
      if ( v12 > 0x80000000 )
        v13 = 0x80000000;
      *v10 = v13;
    }
  }
  sub_1DD9130(a1, v7, 0);
}
