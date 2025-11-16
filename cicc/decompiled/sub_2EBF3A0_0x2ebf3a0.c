// Function: sub_2EBF3A0
// Address: 0x2ebf3a0
//
__int64 __fastcall sub_2EBF3A0(_QWORD *a1, unsigned int a2)
{
  _QWORD *v2; // r13
  __int64 (*v3)(); // rax
  char *v4; // rax
  __int64 v5; // rdx
  unsigned __int16 *v6; // r14
  char *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int16 v10; // r15
  __int64 result; // rax
  __int64 v12; // rax

  v2 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  v3 = *(__int64 (**)())(*v2 + 168LL);
  if ( v3 == sub_2EA3FB0 || (result = ((__int64 (__fastcall *)(_QWORD *, _QWORD))v3)(v2, a2), !(_BYTE)result) )
  {
    v4 = sub_E922F0(v2, a2);
    v6 = (unsigned __int16 *)&v4[2 * v5];
    v7 = v4;
    if ( v4 == (char *)v6 )
    {
      return 1;
    }
    else
    {
      while ( 1 )
      {
        v8 = *(unsigned __int16 *)v7;
        v9 = *(_QWORD *)(a1[38] + 8 * v8);
        v10 = *(_WORD *)v7;
        if ( v9 )
        {
          if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
            break;
          v12 = *(_QWORD *)(v9 + 32);
          if ( v12 )
          {
            if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
              break;
          }
        }
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL))
                                              + 248)
                                  + 16LL)
                      + v8)
          && (*(_QWORD *)(a1[48] + 8 * ((unsigned __int64)v10 >> 6)) & (1LL << v10)) == 0 )
        {
          break;
        }
        v7 += 2;
        if ( v6 == (unsigned __int16 *)v7 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
