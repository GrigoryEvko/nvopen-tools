// Function: sub_AD7E60
// Address: 0xad7e60
//
__int64 __fastcall sub_AD7E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r12
  __int64 v6; // rax
  int v7; // edx
  __int64 v9; // rdx
  int v10; // eax
  int v11; // r14d
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rbx
  char v18; // al
  _BYTE *v19; // rax

  v5 = (_BYTE *)a1;
  if ( *(_BYTE *)a1 != 18 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v10 = *(unsigned __int8 *)(v9 + 8);
    if ( (_BYTE)v10 == 17 )
    {
      v11 = *(_DWORD *)(v9 + 32);
      v12 = 0;
      if ( v11 )
      {
        while ( 1 )
        {
          v13 = sub_AD69F0((unsigned __int8 *)a1, v12);
          v17 = v13;
          if ( !v13 || *(_BYTE *)v13 != 18 )
            break;
          if ( *(_QWORD *)(v13 + 24) == sub_C33340(a1, v12, v14, v15, v16) )
          {
            v18 = *(_BYTE *)(*(_QWORD *)(v17 + 32) + 20LL) & 7;
            if ( v18 == 1 )
              return 0;
          }
          else
          {
            v18 = *(_BYTE *)(v17 + 44) & 7;
            if ( v18 == 1 )
              return 0;
          }
          if ( v18 == 3 || !v18 )
            break;
          if ( v11 == ++v12 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    if ( (unsigned int)(v10 - 17) > 1 )
      return 0;
    a2 = 0;
    v19 = sub_AD7630(a1, 0, v9);
    v5 = v19;
    if ( !v19 || *v19 != 18 )
      return 0;
  }
  v6 = sub_C33340(a1, a2, a3, a4, a5);
  if ( *((_QWORD *)v5 + 3) == v6 )
  {
    v6 = *((_QWORD *)v5 + 4);
    v7 = *(_BYTE *)(v6 + 20) & 7;
    if ( (_BYTE)v7 == 1 )
      return 0;
  }
  else
  {
    v7 = v5[44] & 7;
    if ( (_BYTE)v7 == 1 )
      return 0;
  }
  LOBYTE(v6) = (_BYTE)v7 == 0;
  LOBYTE(v7) = (_BYTE)v7 == 3;
  return (v7 | (unsigned int)v6) ^ 1;
}
