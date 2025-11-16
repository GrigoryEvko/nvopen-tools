// Function: sub_2EAC020
// Address: 0x2eac020
//
__int64 __fastcall sub_2EAC020(__int64 a1, signed int a2, char a3, __int64 a4)
{
  size_t v4; // r8
  unsigned __int8 *v7; // rcx
  int v8; // edx
  __int64 v9; // rdi
  const char *v10; // rax
  size_t v11; // rdx
  signed int v13; // r12d
  size_t v14; // rdx

  v4 = 0;
  v7 = 0;
  if ( a4 )
  {
    v8 = *(_DWORD *)(a4 + 32);
    v9 = *(_QWORD *)(*(_QWORD *)(a4 + 8) + 40LL * (unsigned int)(v8 + a2) + 24);
    if ( a2 >= 0 || a2 < -v8 )
    {
      if ( v9 && (*(_BYTE *)(v9 + 7) & 0x10) != 0 )
      {
        v10 = sub_BD5D20(v9);
        v4 = v11;
        v7 = (unsigned __int8 *)v10;
        a3 = 0;
      }
      else
      {
        a3 = 0;
      }
    }
    else
    {
      v13 = a2;
      a2 += v8;
      if ( v9 && (*(_BYTE *)(v9 + 7) & 0x10) != 0 )
      {
        v7 = (unsigned __int8 *)sub_BD5D20(v9);
        v4 = v14;
        a2 = v13 + *(_DWORD *)(a4 + 32);
      }
      a3 = 1;
    }
  }
  return sub_2EABED0(a1, a2, a3, v7, v4);
}
