// Function: sub_228A980
// Address: 0x228a980
//
__int64 __fastcall sub_228A980(__int64 a1, __int64 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // r14
  bool v7; // zf
  __int64 v8; // r15
  char *v9; // rcx
  char v10; // dl
  char v11; // al
  __int64 v12; // rsi

  LOBYTE(v3) = sub_228A550(a1);
  v4 = v3;
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = 0;
    v7 = *(_WORD *)(a1 + 40) == 0;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v5;
    if ( !v7 )
    {
      do
      {
        v8 = 16 * v6;
        v9 = (char *)(16 * v6 + *(_QWORD *)(a1 + 48));
        v10 = *v9;
        v11 = *v9 & 2;
        if ( (*v9 & 1) != 0 )
          v11 = *v9 & 2 | 4;
        if ( (v10 & 4) != 0 )
          v11 |= 1u;
        *v9 = v10 & 0xF8 | v11 & 7;
        v12 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + v8 + 8);
        if ( v12 )
          *(_QWORD *)(*(_QWORD *)(a1 + 48) + v8 + 8) = sub_DCAF50(a2, v12, 0);
        ++v6;
      }
      while ( *(unsigned __int16 *)(a1 + 40) >= (unsigned int)(v6 + 1) );
    }
  }
  return v4;
}
