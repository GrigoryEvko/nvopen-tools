// Function: sub_35BA000
// Address: 0x35ba000
//
void __fastcall sub_35BA000(__int64 a1)
{
  __int64 v2; // r12
  int v3; // r14d
  __int64 v4; // r13
  int v5; // r14d
  __int64 v6; // r15
  int v7; // eax
  int v8; // r8d
  unsigned int i; // ecx
  __int64 *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  float *v13; // rdx
  float *v14; // rax
  float *v15; // rdi
  unsigned int v16; // ecx
  unsigned __int64 v17; // rdi
  volatile signed __int32 *v18; // rdi
  signed __int32 v19; // eax
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_DWORD *)(v2 + 24);
  if ( v3 )
  {
    v4 = a1 + 16;
    v5 = v3 - 1;
    v6 = *(_QWORD *)(v2 + 8);
    v20[0] = sub_25FD4A0(*(_QWORD **)(a1 + 48), *(_QWORD *)(a1 + 48) + 4LL * *(unsigned int *)(a1 + 40));
    v7 = sub_C4ECF0((int *)(a1 + 40), (__int64 *)v20);
    v8 = 1;
    for ( i = v5 & v7; ; i = v5 & v16 )
    {
      v10 = (__int64 *)(v6 + 8LL * i);
      v11 = *v10;
      if ( v4 == 1 )
      {
        if ( v11 == 1 )
        {
LABEL_13:
          *v10 = 1;
          --*(_DWORD *)(v2 + 16);
          ++*(_DWORD *)(v2 + 20);
          break;
        }
        if ( !v11 )
          break;
      }
      else
      {
        if ( !v11 )
          break;
        if ( v11 != 1 )
        {
          v12 = *(unsigned int *)(a1 + 40);
          if ( (_DWORD)v12 == *(_DWORD *)(v11 + 24) )
          {
            v13 = *(float **)(v11 + 32);
            v14 = *(float **)(a1 + 48);
            v15 = &v14[v12];
            if ( v14 == v15 )
              goto LABEL_13;
            while ( *v14 == *v13 )
            {
              ++v14;
              ++v13;
              if ( v15 == v14 )
                goto LABEL_13;
            }
          }
        }
      }
      v16 = v8 + i;
      ++v8;
    }
  }
  v17 = *(_QWORD *)(a1 + 48);
  if ( v17 )
    j_j___libc_free_0_0(v17);
  v18 = *(volatile signed __int32 **)(a1 + 24);
  if ( v18 )
  {
    if ( &_pthread_key_create )
    {
      v19 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
    }
    else
    {
      v19 = *((_DWORD *)v18 + 3);
      *((_DWORD *)v18 + 3) = v19 - 1;
    }
    if ( v19 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 24LL))(v18);
  }
}
