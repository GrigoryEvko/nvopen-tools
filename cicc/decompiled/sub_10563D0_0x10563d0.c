// Function: sub_10563D0
// Address: 0x10563d0
//
__int64 __fastcall sub_10563D0(__int64 *a1, unsigned __int8 *a2)
{
  int v2; // eax
  __int64 v3; // rcx
  int v4; // edx
  unsigned int v5; // eax
  unsigned __int8 *v6; // rdi
  int v7; // r8d
  __int64 v8; // rdi
  __int64 v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  unsigned int v13; // r8d

  v8 = *a1;
  if ( (unsigned int)*a2 - 30 > 0xA )
  {
    v2 = *(_DWORD *)(v8 + 264);
    v3 = *(_QWORD *)(v8 + 248);
    if ( v2 )
    {
      v4 = v2 - 1;
      v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v6 = *(unsigned __int8 **)(v3 + 8LL * v5);
      if ( a2 == v6 )
        return 1;
      v7 = 1;
      while ( v6 != (unsigned __int8 *)-4096LL )
      {
        v5 = v4 & (v7 + v5);
        v6 = *(unsigned __int8 **)(v3 + 8LL * v5);
        if ( a2 == v6 )
          return 1;
        ++v7;
      }
    }
    return 0;
  }
  else
  {
    v9 = *((_QWORD *)a2 + 5);
    if ( *(_BYTE *)(v8 + 300) )
    {
      v10 = *(_QWORD **)(v8 + 280);
      v11 = &v10[*(unsigned int *)(v8 + 292)];
      if ( v10 == v11 )
      {
        return 0;
      }
      else
      {
        while ( v9 != *v10 )
        {
          if ( v11 == ++v10 )
            return 0;
        }
        return *(unsigned __int8 *)(v8 + 300);
      }
    }
    else
    {
      LOBYTE(v13) = sub_C8CA60(v8 + 272, v9) != 0;
      return v13;
    }
  }
}
