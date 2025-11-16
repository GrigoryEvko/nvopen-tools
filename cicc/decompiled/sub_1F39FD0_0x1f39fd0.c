// Function: sub_1F39FD0
// Address: 0x1f39fd0
//
__int64 __fastcall sub_1F39FD0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  int v5; // esi
  int v6; // r8d
  int v7; // edi
  __int64 v8; // rdx
  int v9; // ecx
  __int64 result; // rax
  unsigned __int64 v11; // r9
  __int64 v12; // rax

  v3 = *(__int64 (**)())(**(_QWORD **)(sub_1E15F70(a2) + 16) + 48LL);
  if ( v3 == sub_1D90020 )
    BUG();
  v4 = v3();
  v5 = *(_DWORD *)(a1 + 36);
  v6 = *(_DWORD *)(a1 + 40);
  v7 = *(_DWORD *)(v4 + 8);
  v8 = v4;
  v9 = **(unsigned __int16 **)(a2 + 16);
  if ( v5 == v9 || (result = 0, v6 == v9) )
  {
    v11 = *(unsigned int *)(v8 + 12);
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
    if ( (int)v12 < 0 )
    {
      result = -((_DWORD)v11 * (unsigned int)((v11 + -*(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL) - 1) / v11));
      if ( v7 != 1 )
      {
LABEL_6:
        if ( v5 != v9 )
          return result;
        return (unsigned int)-(int)result;
      }
    }
    else
    {
      result = (unsigned int)v11 * (unsigned int)((v11 + (int)v12 - 1) / v11);
      if ( v7 != 1 )
        goto LABEL_6;
    }
    if ( v6 == v9 )
      return (unsigned int)-(int)result;
  }
  return result;
}
