// Function: sub_3363B70
// Address: 0x3363b70
//
void __fastcall sub_3363B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  unsigned __int64 v9; // r12
  unsigned __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a2 + 120);
  v7 = v6 + 16LL * *(unsigned int *)(a2 + 128);
  if ( v7 != v6 )
  {
    v8 = a2;
    do
    {
      v9 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
      --*(_DWORD *)(v9 + 216);
      v10[0] = v9;
      if ( (*(_BYTE *)(v8 + 254) & 1) == 0 )
        sub_2F8F5D0(v8, (_QWORD *)a2, a3, a4, a5, a6);
      a2 = (unsigned int)(*(_DWORD *)(v8 + 240) + *(_DWORD *)(v6 + 12));
      sub_2F8F720(v9, (_QWORD *)a2, a3, a4, a5, a6);
      a3 = v10[0];
      if ( !*(_DWORD *)(v10[0] + 216) && v10[0] != a1 + 328 )
      {
        a2 = *(_QWORD *)(a1 + 648);
        if ( a2 == *(_QWORD *)(a1 + 656) )
        {
          sub_2ECAD30(a1 + 640, (_BYTE *)a2, v10);
        }
        else
        {
          if ( a2 )
          {
            *(_QWORD *)a2 = v10[0];
            a2 = *(_QWORD *)(a1 + 648);
          }
          a2 += 8;
          *(_QWORD *)(a1 + 648) = a2;
        }
      }
      v6 += 16;
    }
    while ( v7 != v6 );
  }
}
