// Function: sub_1D49010
// Address: 0x1d49010
//
void __fastcall sub_1D49010(__int64 a1)
{
  _QWORD *v1; // rdi
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r12
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  _QWORD *v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  _QWORD v12[8]; // [rsp+10h] [rbp-40h] BYREF

  v10 = v12;
  v12[0] = a1;
  v1 = v12;
  v11 = 0x400000001LL;
  v2 = 1;
  do
  {
    v3 = v2--;
    v4 = v1[v3 - 1];
    LODWORD(v11) = v2;
    v5 = *(_QWORD *)(v4 + 48);
    if ( v5 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)(v5 + 16);
          if ( *(int *)(v6 + 28) > 0 )
            break;
          v5 = *(_QWORD *)(v5 + 32);
          if ( !v5 )
            goto LABEL_9;
        }
        sub_1D49000(*(_QWORD *)(v5 + 16));
        v9 = (unsigned int)v11;
        if ( (unsigned int)v11 >= HIDWORD(v11) )
        {
          sub_16CD150((__int64)&v10, v12, 0, 8, v7, v8);
          v9 = (unsigned int)v11;
        }
        v10[v9] = v6;
        v5 = *(_QWORD *)(v5 + 32);
        LODWORD(v11) = v11 + 1;
      }
      while ( v5 );
LABEL_9:
      v2 = v11;
      v1 = v10;
    }
  }
  while ( v2 );
  if ( v1 != v12 )
    _libc_free((unsigned __int64)v1);
}
