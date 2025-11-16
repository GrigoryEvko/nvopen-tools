// Function: sub_321AE90
// Address: 0x321ae90
//
void __fastcall sub_321AE90(__int64 a1)
{
  char v1; // r12
  __int64 v2; // rax
  unsigned __int64 *v3; // r14
  unsigned __int64 *v4; // r13
  unsigned __int64 *v5; // r12

  if ( !*(_QWORD *)(a1 + 104) )
  {
    v1 = *(_BYTE *)(*(_QWORD *)(a1 + 112) + 24LL);
    v2 = sub_22077B0(0x70u);
    if ( v2 )
    {
      *(_QWORD *)(v2 + 8) = 0;
      *(_QWORD *)v2 = v2 + 24;
      *(_QWORD *)(v2 + 16) = 32;
      *(_QWORD *)(v2 + 56) = 0;
      *(_QWORD *)(v2 + 80) = &unk_4A35738;
      *(_QWORD *)(v2 + 64) = 0;
      *(_QWORD *)(v2 + 72) = 0;
      *(_QWORD *)(v2 + 88) = v2;
      *(_QWORD *)(v2 + 96) = v2 + 56;
      *(_BYTE *)(v2 + 104) = v1;
    }
    v3 = *(unsigned __int64 **)(a1 + 104);
    *(_QWORD *)(a1 + 104) = v2;
    if ( v3 )
    {
      v4 = (unsigned __int64 *)v3[8];
      v5 = (unsigned __int64 *)v3[7];
      if ( v4 != v5 )
      {
        do
        {
          if ( (unsigned __int64 *)*v5 != v5 + 2 )
            j_j___libc_free_0(*v5);
          v5 += 4;
        }
        while ( v4 != v5 );
        v5 = (unsigned __int64 *)v3[7];
      }
      if ( v5 )
        j_j___libc_free_0((unsigned __int64)v5);
      if ( (unsigned __int64 *)*v3 != v3 + 3 )
        _libc_free(*v3);
      j_j___libc_free_0((unsigned __int64)v3);
    }
  }
  *(_BYTE *)(a1 + 120) = 1;
}
