// Function: sub_2AA9740
// Address: 0x2aa9740
//
void __fastcall sub_2AA9740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // r13
  _QWORD *v13; // [rsp+0h] [rbp-240h] BYREF
  __int64 v14; // [rsp+8h] [rbp-238h]
  _QWORD v15[70]; // [rsp+10h] [rbp-230h] BYREF

  v13 = v15;
  v15[0] = a1;
  v6 = v15;
  v14 = 0x4000000001LL;
  LODWORD(v7) = 1;
  do
  {
    v8 = (unsigned int)v7;
    v7 = (unsigned int)(v7 - 1);
    v9 = v6[v8 - 1];
    LODWORD(v14) = v7;
    v10 = *(__int64 **)(v9 + 24);
    *(_DWORD *)(v9 + 16) = *(_DWORD *)(*(_QWORD *)(v9 + 8) + 16LL) + 1;
    v11 = &v10[*(unsigned int *)(v9 + 32)];
    if ( v10 != v11 )
    {
      do
      {
        v12 = *v10;
        if ( *(_DWORD *)(*v10 + 16) != *(_DWORD *)(*(_QWORD *)(*v10 + 8) + 16LL) + 1 )
        {
          if ( v7 + 1 > (unsigned __int64)HIDWORD(v14) )
          {
            sub_C8D5F0((__int64)&v13, v15, v7 + 1, 8u, a5, a6);
            v7 = (unsigned int)v14;
          }
          v13[v7] = v12;
          v7 = (unsigned int)(v14 + 1);
          LODWORD(v14) = v14 + 1;
        }
        ++v10;
      }
      while ( v11 != v10 );
      v6 = v13;
    }
  }
  while ( (_DWORD)v7 );
  if ( v6 != v15 )
    _libc_free((unsigned __int64)v6);
}
