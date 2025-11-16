// Function: sub_2FDCF80
// Address: 0x2fdcf80
//
__int64 __fastcall sub_2FDCF80(__int64 a1, __int64 a2)
{
  __int64 (*v4)(); // rax
  __int64 v5; // rax
  int v6; // edi
  int v7; // esi
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rcx
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r10

  v4 = *(__int64 (**)())(**(_QWORD **)(sub_2E88D60(a2) + 16) + 136LL);
  if ( v4 == sub_2DD19D0 )
    BUG();
  v5 = v4();
  v6 = *(_DWORD *)(a1 + 64);
  v7 = *(unsigned __int16 *)(a2 + 68);
  v8 = *(_DWORD *)(v5 + 8);
  v9 = *(_DWORD *)(a1 + 68);
  v10 = v5;
  if ( v6 == v7 || (result = 0, v9 == v7) )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
    v13 = 1LL << *(_BYTE *)(v10 + 12);
    v14 = -v13;
    if ( (int)v12 < 0 )
    {
      result = -((unsigned int)v14 & ((_DWORD)v13 - (_DWORD)v12 - 1));
      if ( v8 != 1 )
      {
LABEL_6:
        if ( v6 != v7 )
          return result;
        return (unsigned int)-(int)result;
      }
    }
    else
    {
      result = (unsigned int)v14 & ((_DWORD)v13 + (_DWORD)v12 - 1);
      if ( v8 != 1 )
        goto LABEL_6;
    }
    if ( v9 == v7 )
      return (unsigned int)-(int)result;
  }
  return result;
}
