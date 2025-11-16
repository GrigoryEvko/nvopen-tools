// Function: sub_2042650
// Address: 0x2042650
//
__int64 __fastcall sub_2042650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // r13d
  char v7; // al
  int v8; // r13d
  int v9; // r13d
  int v10; // r13d
  __int64 v11; // rdx
  int v12; // eax
  int v13; // eax
  unsigned int *v14; // rax
  int v16; // r13d

  v6 = 1;
  v7 = *(_BYTE *)(a2 + 229);
  if ( (v7 & 4) == 0 )
  {
    v8 = -((v7 & 8) == 0);
    LOBYTE(v8) = v8 & 0x38;
    v9 = v8 + 201;
    if ( *(_DWORD *)(a1 + 196) > dword_4FCECC0 )
    {
      if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
        sub_1F01F70(a2, (_QWORD *)(unsigned int)dword_4FCECC0, a3, a4, a5, a6);
      v16 = v9 + 10 * *(_DWORD *)(a2 + 244);
      if ( (unsigned __int8)sub_20421A0((_QWORD *)a1, (__int64 *)a2) )
        v16 *= 4;
      v6 = v16 - 20 * sub_2042530((_QWORD *)a1, (__int64 *)a2, 1);
    }
    else
    {
      if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
        sub_1F01F70(a2, (_QWORD *)(unsigned int)dword_4FCECC0, a3, a4, a5, a6);
      v10 = v9 + 10 * (*(_DWORD *)(*(_QWORD *)(a1 + 24) + 4LL * *(unsigned int *)(a2 + 192)) + *(_DWORD *)(a2 + 244));
      if ( (unsigned __int8)sub_20421A0((_QWORD *)a1, (__int64 *)a2) )
        v10 *= 4;
      v6 = v10 - 10 * sub_2042530((_QWORD *)a1, (__int64 *)a2, 0);
    }
    v11 = *(_QWORD *)a2;
    do
    {
      if ( !v11 )
        break;
      v12 = *(__int16 *)(v11 + 24);
      if ( *(__int16 *)(v11 + 24) < 0 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) + ((__int64)~v12 << 6) + 8) & 0x10) != 0 )
          v6 += 5 * *(_DWORD *)(v11 + 60) + 50;
      }
      else if ( (__int16)v12 > 47 )
      {
        if ( (_WORD)v12 == 193 )
          v6 += 15;
      }
      else if ( (__int16)v12 > 45 || (_WORD)v12 == 2 )
      {
        v6 += 5;
      }
      v13 = *(_DWORD *)(v11 + 56);
      if ( !v13 )
        break;
      v14 = (unsigned int *)(*(_QWORD *)(v11 + 32) + 40LL * (unsigned int)(v13 - 1));
      v11 = *(_QWORD *)v14;
    }
    while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v14 + 40LL) + 16LL * v14[2]) == 111 );
  }
  return v6;
}
