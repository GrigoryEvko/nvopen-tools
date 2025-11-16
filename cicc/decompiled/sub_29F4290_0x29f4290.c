// Function: sub_29F4290
// Address: 0x29f4290
//
void __fastcall sub_29F4290(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  const char *v5; // rax
  size_t v6; // rdx
  size_t v7; // r14
  const char *v8; // rax
  size_t v9; // rdx
  size_t v10; // rcx
  bool v11; // cc
  size_t v12; // rdx
  int v13; // eax
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  size_t v21; // [rsp+10h] [rbp-40h]
  const char *s2; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 2 )
  {
    v3 = a1 + 4;
    do
    {
      v4 = v3;
      v5 = sub_BD5D20(a1[1]);
      v7 = v6;
      s2 = v5;
      v8 = sub_BD5D20(*(v3 - 1));
      v10 = v9;
      v11 = v9 <= v7;
      v12 = v7;
      if ( v11 )
        v12 = v10;
      if ( v12 && (v21 = v10, v13 = memcmp(v8, s2, v12), v10 = v21, v13) )
      {
        if ( v13 < 0 )
          goto LABEL_10;
      }
      else if ( v10 != v7 && v10 < v7 )
      {
LABEL_10:
        v14 = (char *)(v3 - 2) - (char *)a1;
        v15 = *(v3 - 2);
        v16 = *(v3 - 1);
        v17 = v3;
        v18 = v14 >> 4;
        if ( v14 > 0 )
        {
          do
          {
            v19 = *(v17 - 4);
            v17 -= 2;
            *v17 = v19;
            v17[1] = *(v17 - 1);
            --v18;
          }
          while ( v18 );
        }
        *a1 = v15;
        a1[1] = v16;
        goto LABEL_13;
      }
      sub_29F41F0(v3 - 2);
LABEL_13:
      v3 += 2;
    }
    while ( v4 != a2 );
  }
}
