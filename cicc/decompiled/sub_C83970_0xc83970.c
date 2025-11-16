// Function: sub_C83970
// Address: 0xc83970
//
void __fastcall sub_C83970(char **a1, unsigned int a2)
{
  char *v2; // r12
  char *v3; // rax
  char *v5; // r12
  char v6; // r14
  char *v7; // rbx
  char *v8; // rbx
  size_t v9; // rdi
  char *v10; // r8
  size_t v11; // rbx
  char *v12; // r13
  char *v13; // [rsp+8h] [rbp-D8h]
  _BYTE *v14; // [rsp+10h] [rbp-D0h] BYREF
  size_t v15; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v16; // [rsp+20h] [rbp-C0h]
  _BYTE v17[184]; // [rsp+28h] [rbp-B8h] BYREF

  v2 = a1[1];
  if ( v2 )
  {
    v3 = *a1;
    v5 = &v2[(_QWORD)*a1];
    if ( a2 <= 1 )
    {
      while ( v3 != v5 )
      {
        if ( *v3 == 92 )
          *v3 = 47;
        ++v3;
      }
    }
    else
    {
      if ( v3 != v5 )
      {
        v6 = 92;
        v7 = *a1;
        if ( a2 != 3 )
          v6 = 47;
        do
        {
          if ( sub_C80220(*v7, a2) )
            *v7 = v6;
          ++v7;
        }
        while ( v5 != v7 );
        v5 = *a1;
      }
      if ( *v5 == 126 && (a1[1] == (char *)1 || sub_C80220(v5[1], a2)) )
      {
        v15 = 0;
        v14 = v17;
        v16 = 128;
        sub_C83840(&v14);
        v8 = a1[1];
        v9 = v15;
        v10 = &v8[(_QWORD)*a1];
        v11 = (size_t)(v8 - 1);
        v12 = *a1 + 1;
        if ( v11 + v15 > v16 )
        {
          v13 = v10;
          sub_C8D290(&v14, v17, v11 + v15, 1);
          v9 = v15;
          v10 = v13;
        }
        if ( v12 != v10 )
        {
          memcpy(&v14[v9], v12, v11);
          v9 = v15;
        }
        v15 = v9 + v11;
        sub_C7FCA0((__int64)a1, (__int64)&v14);
        if ( v14 != v17 )
          _libc_free(v14, &v14);
      }
    }
  }
}
