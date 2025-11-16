// Function: sub_697930
// Address: 0x697930
//
__int64 __fastcall sub_697930(__int64 a1, unsigned int a2, int a3, int a4, int a5, _DWORD *a6, __int64 a7, _DWORD *a8)
{
  __int64 v8; // rax
  int i; // ebx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v18; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v19[208]; // [rsp+20h] [rbp-D0h] BYREF

  v8 = a1;
  for ( i = (int)a6; *(_BYTE *)(v8 + 140) == 12; v8 = *(_QWORD *)(v8 + 160) )
    ;
  v11 = *(_QWORD *)(*(_QWORD *)v8 + 96LL);
  *a6 = 0;
  if ( a8 )
  {
    *a8 = 0;
    v12 = *(_QWORD *)(v11 + 8);
    if ( v12 )
    {
      sub_6E1DD0(&v18);
      sub_6E1E00(4, v19, 0, 1);
      v12 = sub_836920(a1, a2, a3, a4, a5, i, a7);
      sub_6E2B30(a1, a2);
      sub_6E1DF0(v18);
      if ( v12 )
        *a8 = (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 194LL) & 2) != 0;
    }
    else
    {
      *a8 = (*(_BYTE *)(v11 + 176) & 1) == 0;
    }
  }
  else
  {
    v12 = *(_QWORD *)(v11 + 8);
    if ( v12 )
    {
      sub_6E1DD0(&v18);
      sub_6E1E00(4, v19, 0, 1);
      v12 = sub_836920(a1, a2, a3, a4, a5, i, a7);
      sub_6E2B30(a1, a2);
      sub_6E1DF0(v18);
    }
  }
  return v12;
}
