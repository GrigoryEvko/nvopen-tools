// Function: sub_2E5E0A0
// Address: 0x2e5e0a0
//
void __fastcall sub_2E5E0A0(unsigned __int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 v3; // r15
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned __int64 **)(a1 + 280);
  *(_QWORD *)a1 = &unk_4A28C30;
  v9 = *(unsigned __int64 **)(a1 + 288);
  if ( v9 != v2 )
  {
    do
    {
      v3 = *v2;
      if ( *v2 )
      {
        v4 = *(_QWORD *)(v3 + 176);
        if ( v4 != v3 + 192 )
          _libc_free(v4);
        v5 = *(_QWORD *)(v3 + 88);
        if ( v5 != v3 + 104 )
          _libc_free(v5);
        sub_C7D6A0(*(_QWORD *)(v3 + 64), 8LL * *(unsigned int *)(v3 + 80), 8);
        v6 = *(unsigned __int64 **)(v3 + 40);
        v7 = *(unsigned __int64 **)(v3 + 32);
        if ( v6 != v7 )
        {
          do
          {
            if ( *v7 )
              sub_2E5DCD0(*v7);
            ++v7;
          }
          while ( v6 != v7 );
          v7 = *(unsigned __int64 **)(v3 + 32);
        }
        if ( v7 )
          j_j___libc_free_0((unsigned __int64)v7);
        v8 = *(_QWORD *)(v3 + 8);
        if ( v8 != v3 + 24 )
          _libc_free(v8);
        j_j___libc_free_0(v3);
      }
      ++v2;
    }
    while ( v9 != v2 );
    v2 = *(unsigned __int64 **)(a1 + 280);
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 16LL * *(unsigned int *)(a1 + 272), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 16LL * *(unsigned int *)(a1 + 240), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
