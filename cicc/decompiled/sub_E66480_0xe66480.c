// Function: sub_E66480
// Address: 0xe66480
//
__int64 __fastcall sub_E66480(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rcx
  __int64 v4; // rdx
  bool v5; // zf
  char v7; // [rsp-9h] [rbp-9h] BYREF
  __int64 v8; // [rsp-8h] [rbp-8h]

  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 )
  {
    v4 = 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 88);
    if ( !v3 )
      BUG();
    v4 = 1;
  }
  v8 = v2;
  v5 = *(_QWORD *)(a1 + 136) == 0;
  v7 = v4;
  if ( v5 )
    sub_4263D6(a1, a2, v4);
  return (*(__int64 (__fastcall **)(__int64, __int64, char *, __int64, __int64))(a1 + 144))(
           a1 + 120,
           a2,
           &v7,
           v3,
           a1 + 96);
}
