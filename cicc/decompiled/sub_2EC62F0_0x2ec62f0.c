// Function: sub_2EC62F0
// Address: 0x2ec62f0
//
void __fastcall sub_2EC62F0(_QWORD *a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 *v7; // rax
  unsigned __int64 *v8; // r13
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  _QWORD *v11; // rdi

  v5 = a1[114];
  if ( a2 != (unsigned __int64 *)v5 )
  {
    v6 = a1[113];
    if ( a2 == a3 )
      goto LABEL_9;
    if ( !a2 )
      BUG();
    goto LABEL_4;
  }
  if ( !a2 )
    BUG();
  if ( (*(_BYTE *)a2 & 4) == 0 && (*((_BYTE *)a2 + 44) & 8) != 0 )
  {
    do
      v5 = *(_QWORD *)(v5 + 8);
    while ( (*(_BYTE *)(v5 + 44) & 8) != 0 );
  }
  v6 = a1[113];
  a1[114] = *(_QWORD *)(v5 + 8);
  if ( a2 != a3 )
  {
LABEL_4:
    v7 = a2;
    if ( (*(_BYTE *)a2 & 4) == 0 && (*((_BYTE *)a2 + 44) & 8) != 0 )
    {
      do
        v7 = (unsigned __int64 *)v7[1];
      while ( (*((_BYTE *)v7 + 44) & 8) != 0 );
    }
    v8 = (unsigned __int64 *)v7[1];
    if ( a2 != v8 && a3 != v8 )
    {
      sub_2E310C0((__int64 *)(v6 + 40), (__int64 *)(v6 + 40), (__int64)a2, v7[1]);
      if ( v8 != a2 )
      {
        v9 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v8;
        *v8 = *v8 & 7 | *a2 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = *a3;
        *(_QWORD *)(v9 + 8) = a3;
        v10 &= 0xFFFFFFFFFFFFFFF8LL;
        *a2 = v10 | *a2 & 7;
        *(_QWORD *)(v10 + 8) = a2;
        *a3 = v9 | *a3 & 7;
      }
    }
  }
LABEL_9:
  v11 = (_QWORD *)a1[433];
  if ( v11 )
    sub_2E19810(v11, (unsigned __int64)a2, 1);
  if ( (unsigned __int64 *)a1[114] == a3 )
    a1[114] = a2;
}
