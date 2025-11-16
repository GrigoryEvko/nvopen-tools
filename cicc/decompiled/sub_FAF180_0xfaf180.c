// Function: sub_FAF180
// Address: 0xfaf180
//
__int64 __fastcall sub_FAF180(__int64 a1, __int64 **a2, __int64 ***a3)
{
  int v4; // r13d
  unsigned int v5; // r8d
  __int64 v7; // r15
  int v8; // r13d
  __int64 **v9; // r14
  char v10; // al
  int v11; // r14d
  __int64 **v12; // [rsp+10h] [rbp-40h]
  int v13; // [rsp+18h] [rbp-38h]
  unsigned int i; // [rsp+1Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = v4 - 1;
    v13 = 1;
    v12 = 0;
    for ( i = v8 & sub_FAE360(*a2); ; i = v11 )
    {
      v9 = (__int64 **)(v7 + 8LL * i);
      v5 = sub_FAEA90(*a2, *v9);
      if ( (_BYTE)v5 )
      {
        *a3 = v9;
        return v5;
      }
      if ( (unsigned __int8)sub_FAEA90(*v9, (__int64 *)0xFFFFFFFFFFFFF000LL) )
        break;
      v10 = sub_FAEA90(*v9, (__int64 *)0xFFFFFFFFFFFFE000LL);
      if ( !v12 )
      {
        if ( !v10 )
          v9 = 0;
        v12 = v9;
      }
      v11 = v8 & (v13 + i);
      ++v13;
    }
    v5 = 0;
    if ( v12 )
      v9 = v12;
    *a3 = v9;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v5;
}
