// Function: sub_ACF6E0
// Address: 0xacf6e0
//
__int64 __fastcall sub_ACF6E0(__int64 a1)
{
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 i; // rbx
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v11; // [rsp+8h] [rbp-178h]
  int v12; // [rsp+1Ch] [rbp-164h] BYREF
  __int64 v13[4]; // [rsp+20h] [rbp-160h] BYREF
  __int64 *v14; // [rsp+40h] [rbp-140h] BYREF
  __int64 v15; // [rsp+48h] [rbp-138h]
  _BYTE v16[304]; // [rsp+50h] [rbp-130h] BYREF

  v1 = 0;
  v2 = *(_DWORD *)(a1 + 4);
  v15 = 0x2000000000LL;
  v3 = 0;
  v14 = (__int64 *)v16;
  v4 = v2 & 0x7FFFFFF;
  if ( (unsigned int)v4 > 0x20uLL )
  {
    sub_C8D5F0(&v14, v16, (unsigned int)v4, 8);
    v1 = (unsigned int)v15;
    v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    v3 = (unsigned int)v15;
  }
  if ( (_DWORD)v4 )
  {
    v5 = (unsigned int)(v4 - 1);
    for ( i = 0; ; ++i )
    {
      v7 = *(_QWORD *)(a1 + 32 * (i - v4));
      if ( v3 + 1 > (unsigned __int64)HIDWORD(v15) )
      {
        v11 = *(_QWORD *)(a1 + 32 * (i - v4));
        sub_C8D5F0(&v14, v16, v3 + 1, 8);
        v3 = (unsigned int)v15;
        v7 = v11;
      }
      v14[v3] = v7;
      v3 = (unsigned int)(v15 + 1);
      LODWORD(v15) = v15 + 1;
      if ( v5 == i )
        break;
      v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    }
    v1 = (unsigned int)v3;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v13[2] = v1;
  v13[1] = (__int64)v14;
  v13[0] = v8;
  v12 = sub_AC5F60(v14, (__int64)&v14[v1]);
  v9 = sub_AC7800(v13, &v12);
  if ( v14 != (__int64 *)v16 )
    _libc_free(v14, &v12);
  return v9;
}
