// Function: sub_35BA160
// Address: 0x35ba160
//
void __fastcall sub_35BA160(__int64 a1)
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
  float *v12; // rdx
  float *v13; // rax
  float *v14; // rdi
  unsigned int v15; // ecx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  volatile signed __int32 *v19; // rdi
  signed __int32 v20; // eax
  unsigned __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_DWORD *)(v2 + 24);
  if ( v3 )
  {
    v4 = a1 + 16;
    v5 = v3 - 1;
    v6 = *(_QWORD *)(v2 + 8);
    v21[0] = sub_25FD4A0(
               *(_QWORD **)(a1 + 48),
               *(_QWORD *)(a1 + 48) + 4LL * (unsigned int)(*(_DWORD *)(a1 + 44) * *(_DWORD *)(a1 + 40)));
    v7 = sub_35B9BD0((int *)(a1 + 40), (int *)(a1 + 44), (__int64 *)v21);
    v8 = 1;
    for ( i = v5 & v7; ; i = v5 & v15 )
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
        if ( v11 != 1 && *(_QWORD *)(a1 + 40) == *(_QWORD *)(v11 + 24) )
        {
          v12 = *(float **)(v11 + 32);
          v13 = *(float **)(a1 + 48);
          v14 = &v13[*(_DWORD *)(a1 + 44) * *(_DWORD *)(a1 + 40)];
          if ( v13 == v14 )
            goto LABEL_13;
          while ( *v13 == *v12 )
          {
            ++v13;
            ++v12;
            if ( v14 == v13 )
              goto LABEL_13;
          }
        }
      }
      v15 = v8 + i;
      ++v8;
    }
  }
  v16 = *(_QWORD *)(a1 + 72);
  if ( v16 )
    j_j___libc_free_0_0(v16);
  v17 = *(_QWORD *)(a1 + 64);
  if ( v17 )
    j_j___libc_free_0_0(v17);
  v18 = *(_QWORD *)(a1 + 48);
  if ( v18 )
    j_j___libc_free_0_0(v18);
  v19 = *(volatile signed __int32 **)(a1 + 24);
  if ( v19 )
  {
    if ( &_pthread_key_create )
    {
      v20 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
    }
    else
    {
      v20 = *((_DWORD *)v19 + 3);
      *((_DWORD *)v19 + 3) = v20 - 1;
    }
    if ( v20 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
  }
}
