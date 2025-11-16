// Function: sub_222BE90
// Address: 0x222be90
//
__int64 __fastcall sub_222BE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 v6; // rdi
  unsigned int v7; // r12d
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v11; // [rsp+8h] [rbp-D0h]
  __int64 v12; // [rsp+18h] [rbp-C0h] BYREF
  char v13[128]; // [rsp+20h] [rbp-B8h] BYREF
  _BYTE v14[56]; // [rsp+A0h] [rbp-38h] BYREF

  v4 = 1;
  if ( *(_QWORD *)(a1 + 32) < *(_QWORD *)(a1 + 40) )
  {
    a2 = 0xFFFFFFFFLL;
    LOBYTE(v4) = (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, 0xFFFFFFFFLL) != -1;
  }
  if ( *(_BYTE *)(a1 + 170) )
  {
    v6 = *(_QWORD *)(a1 + 200);
    if ( !v6 )
      sub_426219(0, a2, a3, a4);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL))(v6);
    if ( !(_BYTE)v7 )
    {
      if ( (_BYTE)v4 )
      {
        while ( 1 )
        {
          v8 = (*(__int64 (__fastcall **)(_QWORD, __int64, char *, _BYTE *, __int64 *))(**(_QWORD **)(a1 + 200) + 24LL))(
                 *(_QWORD *)(a1 + 200),
                 a1 + 132,
                 v13,
                 v14,
                 &v12);
          v9 = v8;
          if ( v8 == 2 )
            break;
          if ( v8 <= 1 && v12 - (__int64)v13 > 0 )
          {
            v11 = v12 - (_QWORD)v13;
            if ( v11 != sub_2207DF0((FILE **)(a1 + 104), v13, v12 - (_QWORD)v13) )
              return v7;
            if ( v9 == 1 )
              continue;
          }
          LOBYTE(v7) = (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, 0xFFFFFFFFLL) != -1;
          return v7;
        }
        return v7;
      }
    }
  }
  return v4;
}
