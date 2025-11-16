// Function: sub_1F70310
// Address: 0x1f70310
//
__int64 __fastcall sub_1F70310(__int64 a1, unsigned int a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  int v4; // eax
  unsigned int v5; // r12d
  int v8; // esi
  __int64 *v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  __int16 v12; // dx

  v3 = a1;
  v4 = *(unsigned __int16 *)(a1 + 24);
  LOBYTE(a1) = v4 == 32 || v4 == 10;
  if ( (_BYTE)a1 )
  {
    return !(((*(_BYTE *)(v3 + 26) & 8) != 0) & a3);
  }
  else
  {
    v5 = a1;
    if ( v4 == 104 )
    {
      v8 = sub_1F701D0(v3, a2);
      v9 = *(__int64 **)(v3 + 32);
      v10 = &v9[5 * *(unsigned int *)(v3 + 56)];
      if ( v9 == v10 )
      {
        return 1;
      }
      else
      {
        while ( 1 )
        {
          v11 = *v9;
          v12 = *(_WORD *)(*v9 + 24);
          if ( v12 != 48
            && (v12 != 32 && v12 != 10
             || v8 != *(_DWORD *)(*(_QWORD *)(v11 + 88) + 32LL)
             || (*(_BYTE *)(v11 + 26) & 8) != 0 && a3) )
          {
            break;
          }
          v9 += 5;
          if ( v10 == v9 )
            return 1;
        }
      }
    }
  }
  return v5;
}
