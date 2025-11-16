// Function: sub_F5A3F0
// Address: 0xf5a3f0
//
__int64 __fastcall sub_F5A3F0(__int64 a1)
{
  unsigned int v1; // r12d
  char *v2; // rax
  char *v3; // r13
  _QWORD *v4; // rdi
  char *v5; // rbx
  char *v7; // rax
  __int64 v8; // [rsp+0h] [rbp-80h] BYREF
  char *v9; // [rsp+8h] [rbp-78h]
  __int64 v10; // [rsp+10h] [rbp-70h]
  int v11; // [rsp+18h] [rbp-68h]
  char v12; // [rsp+1Ch] [rbp-64h]
  char v13; // [rsp+20h] [rbp-60h] BYREF

  v8 = 0;
  v9 = &v13;
  v10 = 8;
  v11 = 0;
  v12 = 1;
  v1 = sub_F59E00(a1, (__int64)&v8);
  v2 = v9;
  if ( v12 )
  {
    v3 = &v9[8 * HIDWORD(v10)];
    if ( v9 != v3 )
      goto LABEL_3;
    return v1;
  }
  else
  {
    v3 = &v9[8 * (unsigned int)v10];
    if ( v9 != v3 )
    {
LABEL_3:
      while ( 1 )
      {
        v4 = *(_QWORD **)v2;
        v5 = v2;
        if ( *(_QWORD *)v2 < 0xFFFFFFFFFFFFFFFELL )
          break;
        v2 += 8;
        if ( v3 == v2 )
          goto LABEL_5;
      }
      if ( v2 == v3 )
      {
LABEL_5:
        if ( !v12 )
          goto LABEL_12;
        return v1;
      }
      do
      {
        sub_B43D60(v4);
        v7 = v5 + 8;
        if ( v5 + 8 == v3 )
          break;
        while ( 1 )
        {
          v4 = *(_QWORD **)v7;
          v5 = v7;
          if ( *(_QWORD *)v7 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v7 += 8;
          if ( v3 == v7 )
            goto LABEL_11;
        }
      }
      while ( v7 != v3 );
LABEL_11:
      if ( v12 )
        return v1;
    }
LABEL_12:
    _libc_free(v9, &v8);
    return v1;
  }
}
