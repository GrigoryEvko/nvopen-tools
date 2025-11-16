// Function: sub_808F30
// Address: 0x808f30
//
__int64 __fastcall sub_808F30(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // r8
  __int128 v4; // rax
  unsigned int v5; // r8d
  int v7; // [rsp-10h] [rbp-10h] BYREF
  int v8; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v9; // [rsp-8h] [rbp-8h]

  if ( a1 == a2 )
    return 0;
  v9 = v2;
  sub_808EC0(a1, &v7);
  *(_QWORD *)&v4 = sub_808EC0(v3, &v8);
  if ( v4 == 0 )
  {
    return 0;
  }
  else
  {
    v5 = 1;
    if ( *((_QWORD *)&v4 + 1) && (_QWORD)v4 && v7 == v8 )
      return (unsigned int)sub_808F30(*(_QWORD *)(*((_QWORD *)&v4 + 1) + 160LL), *(_QWORD *)(v4 + 160));
  }
  return v5;
}
