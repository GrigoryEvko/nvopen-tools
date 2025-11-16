// Function: sub_AA5820
// Address: 0xaa5820
//
unsigned __int64 __fastcall sub_AA5820(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 i; // rbx
  __int64 *v4; // rax
  __int64 *v5; // rdx
  unsigned __int64 v6; // r12
  char v8; // al
  char v9; // dl
  __int64 v10; // [rsp+0h] [rbp-80h] BYREF
  __int64 *v11; // [rsp+8h] [rbp-78h]
  __int64 v12; // [rsp+10h] [rbp-70h]
  int v13; // [rsp+18h] [rbp-68h]
  char v14; // [rsp+1Ch] [rbp-64h]
  __int64 v15; // [rsp+20h] [rbp-60h] BYREF

  v2 = a1;
  v11 = &v15;
  v15 = a1;
  v12 = 0x100000008LL;
  v13 = 0;
  v14 = 1;
  v10 = 1;
  for ( i = sub_AA5780(a1); i; i = sub_AA5780(i) )
  {
    if ( !v14 )
      goto LABEL_9;
    v4 = v11;
    v5 = &v11[HIDWORD(v12)];
    if ( v11 != v5 )
    {
      while ( i != *v4 )
      {
        if ( v5 == ++v4 )
          goto LABEL_14;
      }
      return 0;
    }
LABEL_14:
    if ( HIDWORD(v12) < (unsigned int)v12 )
    {
      ++HIDWORD(v12);
      *v5 = i;
      ++v10;
    }
    else
    {
LABEL_9:
      a2 = i;
      sub_C8CC70(&v10, i);
      v8 = v14;
      if ( !v9 )
      {
        v6 = 0;
        goto LABEL_12;
      }
    }
    v2 = i;
  }
  v6 = sub_AA4F10(v2);
  v8 = v14;
LABEL_12:
  if ( v8 )
    return v6;
  _libc_free(v11, a2);
  return v6;
}
