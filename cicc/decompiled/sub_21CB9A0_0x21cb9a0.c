// Function: sub_21CB9A0
// Address: 0x21cb9a0
//
__int64 __fastcall sub_21CB9A0(__int64 a1, unsigned int a2, _DWORD *a3, unsigned int a4)
{
  int v5; // eax
  unsigned int v6; // r12d
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  unsigned int v14; // eax
  _BYTE v15[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v16; // [rsp+8h] [rbp-28h]

  *a3 = 2;
  v5 = *(unsigned __int16 *)(a1 + 24);
  LOBYTE(a4) = v5 == 148 || v5 == 142;
  v6 = a4;
  if ( !(_BYTE)a4 )
  {
    if ( v5 == 143 )
    {
      v11 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
      v12 = *(_BYTE *)v11;
      v13 = *(_QWORD *)(v11 + 8);
      v15[0] = v12;
      v16 = v13;
      if ( v12 )
        v14 = sub_1F3E310(v15);
      else
        v14 = sub_1F58D40((__int64)v15);
      if ( a2 >= v14 )
      {
        *a3 = 1;
        return 1;
      }
    }
    return v6;
  }
  v7 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
  v8 = *(_BYTE *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v15[0] = v8;
  v16 = v9;
  if ( v8 )
  {
    if ( a2 < (unsigned int)sub_1F3E310(v15) )
      return 0;
  }
  else if ( a2 < (unsigned int)sub_1F58D40((__int64)v15) )
  {
    return 0;
  }
  *a3 = 0;
  return v6;
}
