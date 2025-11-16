// Function: sub_C19340
// Address: 0xc19340
//
void __fastcall sub_C19340(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v6; // rbx
  bool v7; // al
  __int64 v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // [rsp+20h] [rbp-80h]
  char *v14[2]; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v15[104]; // [rsp+38h] [rbp-68h] BYREF

  if ( a1 != a2 )
  {
    v3 = (__int64)(a1 + 9);
    while ( a2 != (_QWORD *)v3 )
    {
      v6 = v3;
      v7 = sub_C185F0(a3, v3, (__int64)a1);
      v3 += 72;
      if ( v7 )
      {
        v13 = *(_QWORD *)(v3 - 72);
        v14[0] = v15;
        v14[1] = (char *)0xC00000000LL;
        if ( *(_DWORD *)(v3 - 56) )
          sub_C15E20((__int64)v14, (char **)(v3 - 64));
        v8 = v3 - 64;
        v9 = v6 - (_QWORD)a1;
        v10 = 0x8E38E38E38E38E39LL * ((v6 - (__int64)a1) >> 3);
        if ( v9 > 0 )
        {
          do
          {
            v11 = *(_QWORD *)(v8 - 80);
            v12 = v8;
            v8 -= 72;
            *(_QWORD *)(v8 + 64) = v11;
            sub_C15E20(v12, (char **)v8);
            --v10;
          }
          while ( v10 );
        }
        *a1 = v13;
        sub_C15E20((__int64)(a1 + 1), v14);
        if ( v14[0] != v15 )
          _libc_free(v14[0], v14);
      }
      else
      {
        sub_C19270(v6, a3);
      }
    }
  }
}
