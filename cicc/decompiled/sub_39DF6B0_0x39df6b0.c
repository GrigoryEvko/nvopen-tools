// Function: sub_39DF6B0
// Address: 0x39df6b0
//
void __fastcall sub_39DF6B0(void *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rdi
  void *v5[2]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v6; // [rsp+10h] [rbp-50h]
  __int64 v7; // [rsp+18h] [rbp-48h]
  int v8; // [rsp+20h] [rbp-40h]
  __int64 v9; // [rsp+28h] [rbp-38h]
  unsigned __int64 v10; // [rsp+30h] [rbp-30h]
  __int64 v11; // [rsp+38h] [rbp-28h]
  unsigned int v12; // [rsp+40h] [rbp-20h]

  v5[0] = a1;
  v5[1] = 0;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  sub_39DEC90(v5, a2);
  if ( v12 )
  {
    v2 = v10;
    v3 = v10 + 48LL * v12;
    do
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)v2 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        {
          v4 = *(_QWORD *)(v2 + 8);
          if ( v4 != v2 + 24 )
            break;
        }
        v2 += 48LL;
        if ( v3 == v2 )
          goto LABEL_7;
      }
      v2 += 48LL;
      j_j___libc_free_0(v4);
    }
    while ( v3 != v2 );
  }
LABEL_7:
  j___libc_free_0(v10);
  j___libc_free_0(v6);
}
