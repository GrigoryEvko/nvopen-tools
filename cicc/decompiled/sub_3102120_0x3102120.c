// Function: sub_3102120
// Address: 0x3102120
//
void __fastcall sub_3102120(unsigned __int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rsi
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rax
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // r14

  v1 = *(unsigned int *)(a1 + 120);
  *(_QWORD *)a1 = &unk_4A32950;
  *(_QWORD *)(a1 + 88) = &unk_4A20C88;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 16 * v1, 8);
  v2 = *(unsigned int *)(a1 + 80);
  *(_QWORD *)(a1 + 48) = &unk_4A20C88;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 16 * v2, 8);
  v3 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A21008;
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 16);
    v5 = &v4[2 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = v4[1];
        if ( v6 )
        {
          if ( (v6 & 4) != 0 )
          {
            v7 = (unsigned __int64 *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
            v8 = (unsigned __int64)v7;
            if ( v7 )
            {
              if ( (unsigned __int64 *)*v7 != v7 + 2 )
                _libc_free(*v7);
              j_j___libc_free_0(v8);
            }
          }
        }
      }
      v4 += 2;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 32);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v3, 8);
  j_j___libc_free_0(a1);
}
