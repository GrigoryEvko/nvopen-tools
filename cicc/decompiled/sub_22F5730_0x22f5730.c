// Function: sub_22F5730
// Address: 0x22f5730
//
__int64 __fastcall sub_22F5730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 result; // rax
  int v8; // esi
  __int64 v9; // rdx
  int v10; // ecx
  int v11; // r8d
  int v12; // ecx

  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 40) = a6;
  *(_QWORD *)a1 = &unk_4A0AA78;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 24) = a4;
  *(_BYTE *)(a1 + 48) = a7;
  *(_WORD *)(a1 + 49) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x300000000LL;
  *(_QWORD *)(a1 + 144) = a1 + 168;
  result = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 8;
  if ( (_DWORD)result )
  {
    v8 = result;
    v9 = *(_QWORD *)(a1 + 32) + 40LL;
    LODWORD(result) = 0;
    do
    {
      while ( 1 )
      {
        v10 = *(unsigned __int8 *)(v9 + 4);
        v11 = result;
        result = (unsigned int)(result + 1);
        if ( v10 != 1 )
          break;
        v12 = *(_DWORD *)v9;
        v9 += 80;
        *(_DWORD *)(a1 + 64) = v12;
        if ( (_DWORD)result == v8 )
          return result;
      }
      if ( v10 == 2 )
      {
        *(_DWORD *)(a1 + 68) = *(_DWORD *)v9;
      }
      else if ( *(_BYTE *)(v9 + 4) )
      {
        *(_DWORD *)(a1 + 72) = v11;
        return result;
      }
      v9 += 80;
    }
    while ( (_DWORD)result != v8 );
  }
  return result;
}
