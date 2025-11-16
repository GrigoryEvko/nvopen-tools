// Function: sub_37B4870
// Address: 0x37b4870
//
__int64 __fastcall sub_37B4870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  v7 = *(_BYTE *)(a2 + 249);
  if ( (v7 & 4) == 0 )
  {
    v8 = -((v7 & 8) == 0);
    LOBYTE(v8) = v8 & 0x38;
    v9 = v8 + 201;
    if ( *(_DWORD *)(a1 + 196) > (int)qword_5051048 )
    {
      if ( (*(_BYTE *)(a2 + 254) & 2) == 0 )
        sub_2F8F770(a2, (_QWORD *)a2, a3, a4, a5, a6);
      v16 = v9 + 10 * *(_DWORD *)(a2 + 244);
      if ( (unsigned __int8)sub_37B43A0((_QWORD *)a1, (__int64 *)a2) )
        v16 *= 4;
      v6 = v16 - 20 * sub_37B4750((_QWORD *)a1, (__int64 *)a2, 1);
    }
    else
    {
      if ( (*(_BYTE *)(a2 + 254) & 2) == 0 )
        sub_2F8F770(a2, (_QWORD *)a2, a3, a4, a5, a6);
      v10 = v9 + 10 * (*(_DWORD *)(*(_QWORD *)(a1 + 24) + 4LL * *(unsigned int *)(a2 + 200)) + *(_DWORD *)(a2 + 244));
      if ( (unsigned __int8)sub_37B43A0((_QWORD *)a1, (__int64 *)a2) )
        v10 *= 4;
      v6 = v10 - 10 * sub_37B4750((_QWORD *)a1, (__int64 *)a2, 0);
    }
    v11 = *(_QWORD *)a2;
    do
    {
      if ( !v11 )
        break;
      v12 = *(_DWORD *)(v11 + 24);
      if ( v12 < 0 )
      {
        if ( *(char *)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) - 40LL * (unsigned int)~v12 + 24) < 0 )
          v6 += 5 * *(_DWORD *)(v11 + 68) + 50;
      }
      else if ( v12 > 50 )
      {
        if ( (unsigned int)(v12 - 307) < 2 )
          v6 += 15;
      }
      else if ( v12 > 48 || v12 == 2 )
      {
        v6 += 5;
      }
      v13 = *(_DWORD *)(v11 + 64);
      if ( !v13 )
        break;
      v14 = (unsigned int *)(*(_QWORD *)(v11 + 40) + 40LL * (unsigned int)(v13 - 1));
      v11 = *(_QWORD *)v14;
    }
    while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v14 + 48LL) + 16LL * v14[2]) == 262 );
  }
  return v6;
}
