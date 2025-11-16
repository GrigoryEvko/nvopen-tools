// Function: sub_1AFB750
// Address: 0x1afb750
//
void __fastcall sub_1AFB750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // r14
  __int64 *v11; // r12
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
    v10 = *(__int64 **)(v9 + 32);
    v11 = *(__int64 **)(v9 + 24);
    *(_DWORD *)(v9 + 16) = *(_DWORD *)(*(_QWORD *)(v9 + 8) + 16LL) + 1;
    if ( v11 != v10 )
    {
      do
      {
        v12 = *v11;
        if ( *(_DWORD *)(*v11 + 16) != *(_DWORD *)(*(_QWORD *)(*v11 + 8) + 16LL) + 1 )
        {
          if ( (unsigned int)v7 >= HIDWORD(v14) )
          {
            sub_16CD150((__int64)&v13, v15, 0, 8, a5, a6);
            v7 = (unsigned int)v14;
          }
          v13[v7] = v12;
          v7 = (unsigned int)(v14 + 1);
          LODWORD(v14) = v14 + 1;
        }
        ++v11;
      }
      while ( v10 != v11 );
      v6 = v13;
    }
  }
  while ( (_DWORD)v7 );
  if ( v6 != v15 )
    _libc_free((unsigned __int64)v6);
}
