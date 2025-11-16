// Function: sub_3101EB0
// Address: 0x3101eb0
//
__int64 __fastcall sub_3101EB0(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  unsigned __int64 *v5; // rax
  unsigned __int64 v6; // r13

  v1 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A21008;
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 16);
    v3 = &v2[2 * v1];
    do
    {
      if ( *v2 != -8192 && *v2 != -4096 )
      {
        v4 = v2[1];
        if ( v4 )
        {
          if ( (v4 & 4) != 0 )
          {
            v5 = (unsigned __int64 *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
            v6 = (unsigned __int64)v5;
            if ( v5 )
            {
              if ( (unsigned __int64 *)*v5 != v5 + 2 )
                _libc_free(*v5);
              j_j___libc_free_0(v6);
            }
          }
        }
      }
      v2 += 2;
    }
    while ( v3 != v2 );
    v1 = *(unsigned int *)(a1 + 32);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v1, 8);
}
