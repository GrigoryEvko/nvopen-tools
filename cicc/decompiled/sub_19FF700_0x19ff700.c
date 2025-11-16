// Function: sub_19FF700
// Address: 0x19ff700
//
__int64 __fastcall sub_19FF700(int a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r12d
  __int64 v5; // r13
  unsigned int v6; // r15d
  _QWORD *v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 ***v11; // r14
  __int64 v13; // rbx
  __int64 v14; // r14
  size_t v15; // r14
  _QWORD *v16; // rax
  size_t v17; // rdx
  __int64 **v18; // rdi
  unsigned int v19; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v19 = a1 - 26;
  if ( !v4 )
    return 0;
  v5 = a2;
  v6 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v8 = 16LL * v6;
      if ( sub_15FB730(*(_QWORD *)(*(_QWORD *)v5 + v8 + 8), a2, a3, a4) )
      {
        v10 = sub_15FB7C0(*(_QWORD *)(*(_QWORD *)v5 + v8 + 8), a2, v9, a4);
        a2 = v6;
        v11 = (__int64 ***)v10;
        if ( (unsigned int)sub_19FDED0((__int64 *)v5, v6, v10) != v6 )
        {
          if ( a1 == 26 )
          {
            v18 = *v11;
            return sub_15A06D0(v18, a2, a3, a4);
          }
          if ( a1 == 27 )
            return sub_15A04A0(*v11);
        }
      }
      a3 = *(unsigned int *)(v5 + 8);
      if ( v6 + 1 != (_DWORD)a3 )
      {
        a2 = *(_QWORD *)v5;
        v7 = (_QWORD *)(*(_QWORD *)v5 + v8);
        a4 = v7[1];
        if ( *(_QWORD *)(*(_QWORD *)v5 + 16LL * (v6 + 1) + 8) == a4 )
          break;
      }
      ++v6;
LABEL_6:
      if ( v4 == v6 )
        return 0;
    }
    if ( v19 <= 1 )
    {
      a2 += 16LL * (unsigned int)a3;
      if ( (_QWORD *)a2 != v7 + 2 )
      {
        v17 = a2 - (_QWORD)(v7 + 2);
        a2 = (__int64)(v7 + 2);
        memmove(v7, v7 + 2, v17);
        LODWORD(a3) = *(_DWORD *)(v5 + 8);
      }
      a3 = (unsigned int)(a3 - 1);
      --v4;
      *(_DWORD *)(v5 + 8) = a3;
      goto LABEL_6;
    }
    if ( v4 == 2 )
      break;
    v13 = v8 + 32;
    v14 = 16 * a3;
    a3 = a2 + 16 * a3;
    v15 = v14 - v13;
    if ( a2 + v13 != a3 )
    {
      v16 = memmove(v7, (const void *)(a2 + v13), v15);
      a2 = *(_QWORD *)v5;
      v7 = v16;
    }
    v4 -= 2;
    *(_DWORD *)(v5 + 8) = (__int64)((__int64)v7 + v15 - a2) >> 4;
    if ( v4 == v6 )
      return 0;
  }
  v18 = **(__int64 ****)(a2 + 8);
  return sub_15A06D0(v18, a2, a3, a4);
}
