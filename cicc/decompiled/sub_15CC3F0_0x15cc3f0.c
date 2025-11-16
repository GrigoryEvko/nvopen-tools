// Function: sub_15CC3F0
// Address: 0x15cc3f0
//
void __fastcall sub_15CC3F0(__int64 a1)
{
  _QWORD *v1; // rdi
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 *v5; // r14
  __int64 *v6; // r12
  __int64 v7; // r13
  _QWORD *v8; // [rsp-248h] [rbp-248h] BYREF
  __int64 v9; // [rsp-240h] [rbp-240h]
  _QWORD v10[71]; // [rsp-238h] [rbp-238h] BYREF

  if ( *(_DWORD *)(a1 + 16) != *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL) + 1 )
  {
    v8 = v10;
    v10[0] = a1;
    v1 = v10;
    v9 = 0x4000000001LL;
    LODWORD(v2) = 1;
    do
    {
      v3 = (unsigned int)v2;
      v2 = (unsigned int)(v2 - 1);
      v4 = v1[v3 - 1];
      LODWORD(v9) = v2;
      v5 = *(__int64 **)(v4 + 32);
      v6 = *(__int64 **)(v4 + 24);
      *(_DWORD *)(v4 + 16) = *(_DWORD *)(*(_QWORD *)(v4 + 8) + 16LL) + 1;
      if ( v5 != v6 )
      {
        do
        {
          v7 = *v6;
          if ( *(_DWORD *)(*v6 + 16) != *(_DWORD *)(*(_QWORD *)(*v6 + 8) + 16LL) + 1 )
          {
            if ( (unsigned int)v2 >= HIDWORD(v9) )
            {
              sub_16CD150(&v8, v10, 0, 8);
              v2 = (unsigned int)v9;
            }
            v8[v2] = v7;
            v2 = (unsigned int)(v9 + 1);
            LODWORD(v9) = v9 + 1;
          }
          ++v6;
        }
        while ( v5 != v6 );
        v1 = v8;
      }
    }
    while ( (_DWORD)v2 );
    if ( v1 != v10 )
      _libc_free((unsigned __int64)v1);
  }
}
