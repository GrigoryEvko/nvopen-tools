// Function: sub_2E986C0
// Address: 0x2e986c0
//
__int64 __fastcall sub_2E986C0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v4; // r8
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rbx
  int v9; // r15d
  __int64 (*v10)(); // r14
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v14; // [rsp+8h] [rbp-38h]

  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0 )
  {
    v3 = *(_DWORD *)(a1 + 44);
    if ( (v3 & 4) != 0 || (v3 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & 0x100000LL) == 0 )
        goto LABEL_5;
    }
    else if ( !sub_2E88A90(a1, 0x100000, 1) )
    {
      goto LABEL_5;
    }
  }
  LOBYTE(v6) = sub_2E8B090(a1);
  v4 = v6;
  if ( !(_BYTE)v6 && (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != 0 )
  {
    v8 = *(_QWORD *)(a1 + 32);
    v14 = v8 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
    while ( 1 )
    {
      if ( *(_BYTE *)v8 )
      {
        if ( *(_BYTE *)v8 != 1 )
          break;
      }
      else
      {
        v9 = *(_DWORD *)(v8 + 8);
        if ( v9 < 0 )
        {
          v9 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)a2 + 56LL))(
                 a2,
                 (unsigned int)v9,
                 a3,
                 v7,
                 v4);
          if ( v9 < 0 )
            break;
        }
        v10 = *(__int64 (**)())(*(_QWORD *)a2 + 200LL);
        v11 = sub_2E88D60(a1);
        if ( v10 == sub_2E4EE50 )
          break;
        v12 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v10)(a2, (unsigned int)v9, v11);
        v4 = v12;
        if ( !(_BYTE)v12 )
          break;
      }
      v8 += 40;
      if ( v14 == v8 )
        return (unsigned int)v4;
    }
  }
LABEL_5:
  LODWORD(v4) = 0;
  return (unsigned int)v4;
}
