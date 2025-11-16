// Function: sub_3421DB0
// Address: 0x3421db0
//
void __fastcall sub_3421DB0(__int64 a1)
{
  _QWORD *v1; // rdi
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+8h] [rbp-48h]
  _QWORD v13[8]; // [rsp+10h] [rbp-40h] BYREF

  v11 = v13;
  v13[0] = a1;
  v1 = v13;
  v12 = 0x400000001LL;
  v2 = 1;
  do
  {
    v3 = v2--;
    v4 = v1[v3 - 1];
    LODWORD(v12) = v2;
    v5 = *(_QWORD *)(v4 + 56);
    if ( v5 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)(v5 + 16);
          if ( *(int *)(v6 + 36) > 0 )
            break;
          v5 = *(_QWORD *)(v5 + 32);
          if ( !v5 )
            goto LABEL_9;
        }
        sub_3421DA0(*(_QWORD *)(v5 + 16));
        v9 = (unsigned int)v12;
        v10 = (unsigned int)v12 + 1LL;
        if ( v10 > HIDWORD(v12) )
        {
          sub_C8D5F0((__int64)&v11, v13, v10, 8u, v7, v8);
          v9 = (unsigned int)v12;
        }
        v11[v9] = v6;
        v5 = *(_QWORD *)(v5 + 32);
        LODWORD(v12) = v12 + 1;
      }
      while ( v5 );
LABEL_9:
      v2 = v12;
      v1 = v11;
    }
  }
  while ( v2 );
  if ( v1 != v13 )
    _libc_free((unsigned __int64)v1);
}
