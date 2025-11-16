// Function: sub_397CF80
// Address: 0x397cf80
//
__int64 __fastcall sub_397CF80(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx

  v3 = sub_16CE270((__int64 *)a2, *(_QWORD *)(a1 + 8));
  v4 = v3;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a2 + 64);
    if ( v3 > (unsigned __int64)((*(_QWORD *)(a2 + 72) - v5) >> 3) )
    {
      v4 = 0;
    }
    else
    {
      v4 = 0;
      v6 = *(_QWORD *)(v5 + 8LL * (v3 - 1));
      if ( v6 )
      {
        v4 = *(unsigned int *)(v6 + 8);
        v7 = (unsigned int)(*(_DWORD *)(a1 + 48) - 1);
        if ( (unsigned int)v7 >= (unsigned int)v4 )
          v7 = 0;
        if ( (_DWORD)v4 )
        {
          v8 = v7 - v4;
          v4 = 0;
          v9 = *(_QWORD *)(v6 + 8 * v8);
          if ( *(_BYTE *)v9 == 1 )
          {
            v10 = *(_QWORD *)(v9 + 136);
            if ( *(_BYTE *)(v10 + 16) == 13 )
            {
              v4 = *(_QWORD *)(v10 + 24);
              if ( *(_DWORD *)(v10 + 32) > 0x40u )
                v4 = *(_QWORD *)v4;
            }
          }
        }
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, unsigned __int64))(a2 + 88))(a1, *(_QWORD *)(a2 + 96), v4);
}
