// Function: sub_C20C80
// Address: 0xc20c80
//
__int64 __fastcall sub_C20C80(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // r8
  __int64 result; // rax
  __int64 v5; // r12
  __int64 *v6; // r15
  __int64 *v7; // r13
  __int64 i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // r13
  __int64 *v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdi

  v3 = (_QWORD *)a1[53];
  if ( !v3 )
  {
    v3 = (_QWORD *)sub_22077B0(136);
    if ( v3 )
    {
      memset(v3, 0, 0x88u);
      v3[16] = 1;
      v3[7] = v3 + 9;
      v3[8] = 0x400000000LL;
      v3[13] = v3 + 15;
    }
    v5 = a1[53];
    a1[53] = (__int64)v3;
    if ( v5 )
    {
      v6 = *(__int64 **)(v5 + 56);
      v7 = &v6[*(unsigned int *)(v5 + 64)];
      if ( v6 != v7 )
      {
        for ( i = *(_QWORD *)(v5 + 56); ; i = *(_QWORD *)(v5 + 56) )
        {
          v9 = *v6;
          v10 = (unsigned int)(((__int64)v6 - i) >> 3) >> 7;
          a2 = 4096LL << v10;
          if ( v10 >= 0x1E )
            a2 = 0x40000000000LL;
          ++v6;
          sub_C7D6A0(v9, a2, 16);
          if ( v7 == v6 )
            break;
        }
      }
      v11 = *(__int64 **)(v5 + 104);
      v12 = &v11[2 * *(unsigned int *)(v5 + 112)];
      if ( v11 != v12 )
      {
        do
        {
          a2 = v11[1];
          v13 = *v11;
          v11 += 2;
          sub_C7D6A0(v13, a2, 16);
        }
        while ( v12 != v11 );
        v12 = *(__int64 **)(v5 + 104);
      }
      if ( v12 != (__int64 *)(v5 + 120) )
        _libc_free(v12, a2);
      v14 = *(_QWORD *)(v5 + 56);
      if ( v14 != v5 + 72 )
        _libc_free(v14, a2);
      sub_C7D6A0(*(_QWORD *)(v5 + 16), 16LL * *(unsigned int *)(v5 + 32), 8);
      j_j___libc_free_0(v5, 136);
      v3 = (_QWORD *)a1[53];
    }
  }
  result = sub_C1DD70((__int64)v3, a1[26], a1[27] - a1[26]);
  if ( !(_DWORD)result )
  {
    a1[26] = a1[27];
    sub_C1AFD0();
    return 0;
  }
  return result;
}
