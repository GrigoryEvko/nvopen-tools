// Function: sub_2D256D0
// Address: 0x2d256d0
//
void __fastcall sub_2D256D0(__int64 a1, int a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdi
  _QWORD *v5; // rax
  __int64 v6; // rcx
  _QWORD *i; // rdx

  if ( a2 )
  {
    v3 = 4 * a2 / 3u + 1;
    v4 = (((((((((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 4) | ((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 8)
          | ((((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 4)
          | ((v3 | (v3 >> 1)) >> 2)
          | v3
          | (v3 >> 1)) >> 16)
        | ((((((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 4) | ((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 8)
        | ((((v3 | (v3 >> 1)) >> 2) | v3 | (v3 >> 1)) >> 4)
        | ((v3 | (v3 >> 1)) >> 2)
        | v3
        | (v3 >> 1))
       + 1;
    *(_DWORD *)(a1 + 24) = v4;
    v5 = (_QWORD *)sub_C7D670(272 * v4, 8);
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v5;
    for ( i = &v5[34 * v6]; i != v5; v5 += 34 )
    {
      if ( v5 )
        *v5 = -4096;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
  }
}
