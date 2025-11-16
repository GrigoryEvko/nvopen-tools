// Function: sub_2AAA480
// Address: 0x2aaa480
//
void __fastcall sub_2AAA480(_QWORD *a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 **v4; // rbx
  __int64 v5; // r12
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r12
  __int64 *v8; // rdi
  __int64 v9; // rax

  *a1 = &unk_4A231A8;
  v2 = a1[2];
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v2 & 4) != 0 )
  {
    v4 = *(__int64 ***)v3;
    v5 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  }
  else
  {
    v4 = (__int64 **)(a1 + 2);
    v5 = (__int64)(a1 + 3);
    if ( !v3 )
      goto LABEL_3;
  }
  if ( v4 != (__int64 **)v5 )
  {
    do
    {
      v8 = *v4++;
      v9 = *v8;
      v8[6] = 0;
      (*(void (**)(void))(v9 + 8))();
    }
    while ( (__int64 **)v5 != v4 );
    v2 = a1[2];
  }
LABEL_3:
  if ( v2 )
  {
    if ( (v2 & 4) != 0 )
    {
      v6 = (unsigned __int64 *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = (unsigned __int64)v6;
      if ( v6 )
      {
        if ( (unsigned __int64 *)*v6 != v6 + 2 )
          _libc_free(*v6);
        j_j___libc_free_0(v7);
      }
    }
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
