// Function: sub_1423BB0
// Address: 0x1423bb0
//
__int64 __fastcall sub_1423BB0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rdx
  __int64 *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  _QWORD *v8; // rdx
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rax
  _QWORD *v11; // r12
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax

  v2 = *(unsigned __int8 *)a1;
  if ( (_BYTE)v2 == *(_BYTE *)a2 )
  {
    v3 = a1[1];
    if ( (_BYTE)v2 )
    {
      v5 = (__int64 *)((v3 & 0xFFFFFFFFFFFFFFF8LL) - 72);
      if ( (v3 & 4) != 0 )
        v5 = (__int64 *)((v3 & 0xFFFFFFFFFFFFFFF8LL) - 24);
      v6 = *(_QWORD *)(a2 + 8);
      v7 = *v5;
      v8 = (_QWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) - 72);
      if ( (v6 & 4) != 0 )
        v8 = (_QWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) - 24);
      if ( *v8 == v7 )
      {
        v9 = 0xAAAAAAAAAAAAAAABLL
           * ((__int64)(sub_134EF80(a1 + 1)
                      - ((a1[1] & 0xFFFFFFFFFFFFFFF8LL)
                       - 24LL * (*(_DWORD *)((a1[1] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
        v10 = sub_134EF80((_QWORD *)(a2 + 8));
        v11 = (_QWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL)
                       - 24LL * (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
        if ( (_DWORD)v9 == -1431655765 * (unsigned int)((__int64)(v10 - (_QWORD)v11) >> 3) )
        {
          v12 = sub_134EF80(a1 + 1);
          v13 = (_QWORD *)((a1[1] & 0xFFFFFFFFFFFFFFF8LL)
                         - 24LL * (*(_DWORD *)((a1[1] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
          if ( (_QWORD *)v12 == v13 )
            return v2;
          while ( *v13 == *v11 )
          {
            v13 += 3;
            v11 += 3;
            if ( (_QWORD *)v12 == v13 )
              return v2;
          }
        }
      }
    }
    else if ( *(_QWORD *)(a2 + 8) == v3
           && a1[2] == *(_QWORD *)(a2 + 16)
           && a1[3] == *(_QWORD *)(a2 + 24)
           && a1[4] == *(_QWORD *)(a2 + 32) )
    {
      LOBYTE(v2) = a1[5] == *(_QWORD *)(a2 + 40);
      return v2;
    }
  }
  return 0;
}
