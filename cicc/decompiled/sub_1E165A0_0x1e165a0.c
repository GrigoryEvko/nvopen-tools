// Function: sub_1E165A0
// Address: 0x1e165a0
//
__int64 __fastcall sub_1E165A0(__int64 a1, int a2, char a3, __int64 a4)
{
  int v4; // r10d
  __int64 v6; // r13
  unsigned int v8; // edx
  __int64 v10; // rax
  int v11; // esi
  __int16 *v13; // r15
  __int16 v14; // cx
  __int16 *v15; // r15
  unsigned __int16 v16; // r8
  __int16 v17; // r9
  __int16 *v18; // rcx

  v4 = *(_DWORD *)(a1 + 40);
  if ( !v4 )
    return 0xFFFFFFFFLL;
  v6 = 24LL * (unsigned int)a2;
  v8 = 0;
  v10 = *(_QWORD *)(a1 + 32);
  do
  {
    if ( !*(_BYTE *)v10 && (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
    {
      v11 = *(_DWORD *)(v10 + 8);
      if ( v11 )
      {
        if ( a2 == v11 )
        {
LABEL_11:
          if ( !a3 || (*(_BYTE *)(v10 + 3) & 0x40) != 0 )
            return v8;
        }
        else if ( a4 && v11 > 0 && a2 > 0 )
        {
          v13 = (__int16 *)(*(_QWORD *)(a4 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(a4 + 8) + v6 + 8));
          v14 = *v13;
          v15 = v13 + 1;
          v16 = v14 + a2;
          if ( !v14 )
            v15 = 0;
LABEL_19:
          v18 = v15;
          while ( v18 )
          {
            if ( v11 == v16 )
              goto LABEL_11;
            v17 = *v18;
            v15 = 0;
            ++v18;
            v16 += v17;
            if ( !v17 )
              goto LABEL_19;
          }
        }
      }
    }
    ++v8;
    v10 += 40;
  }
  while ( v4 != v8 );
  return 0xFFFFFFFFLL;
}
