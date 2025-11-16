// Function: sub_1E1CC30
// Address: 0x1e1cc30
//
__int64 __fastcall sub_1E1CC30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int16 v4; // dx
  __int64 v5; // r8
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rbx
  int v11; // r15d
  __int64 (*v12)(); // r14
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_WORD *)v3 != 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0 )
  {
    v4 = *(_WORD *)(a1 + 46);
    if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
    {
      if ( (*(_QWORD *)(v3 + 8) & 0x20000LL) == 0 )
        goto LABEL_5;
    }
    else if ( !sub_1E15D00(a1, 0x20000u, 1) )
    {
      goto LABEL_5;
    }
  }
  LOBYTE(v7) = sub_1E17880(a1);
  v5 = v7;
  if ( !(_BYTE)v7 )
  {
    v9 = *(unsigned int *)(a1 + 40);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD *)(a1 + 32);
      v16 = v10 + 40 * v9;
      while ( 1 )
      {
        if ( *(_BYTE *)v10 )
        {
          if ( *(_BYTE *)v10 != 1 )
            break;
        }
        else
        {
          v11 = *(_DWORD *)(v10 + 8);
          if ( v11 < 0 )
          {
            v11 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(
                    a2,
                    (unsigned int)v11,
                    a3,
                    v8,
                    v5);
            if ( v11 < 0 )
              break;
          }
          v12 = *(__int64 (**)())(*(_QWORD *)a2 + 80LL);
          v13 = sub_1E15F70(a1);
          if ( v12 == sub_1E1C7F0 )
            break;
          v14 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v12)(a2, (unsigned int)v11, v13);
          v5 = v14;
          if ( !(_BYTE)v14 )
            break;
        }
        v10 += 40;
        if ( v16 == v10 )
          return (unsigned int)v5;
      }
    }
  }
LABEL_5:
  LODWORD(v5) = 0;
  return (unsigned int)v5;
}
