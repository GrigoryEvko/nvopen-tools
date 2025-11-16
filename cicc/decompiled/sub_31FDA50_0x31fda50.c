// Function: sub_31FDA50
// Address: 0x31fda50
//
__int64 __fastcall sub_31FDA50(_QWORD *a1, __int64 a2, int a3)
{
  unsigned __int64 v3; // r14
  __int64 result; // rax
  unsigned __int64 *v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // eax
  int v10; // edx
  __int64 v11; // rax
  _DWORD v12[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v13; // [rsp+10h] [rbp-40h] BYREF
  int v14; // [rsp+12h] [rbp-3Eh]
  unsigned __int64 *v15; // [rsp+18h] [rbp-38h]
  __int64 v16; // [rsp+20h] [rbp-30h]

  v3 = a2;
  result = sub_AF18C0(a2);
  if ( (unsigned __int16)result <= 0x17u )
  {
    result = 1LL << result;
    if ( (result & 0x880014) != 0 )
    {
      if ( *(_BYTE *)a2 != 16 )
      {
        result = *(unsigned __int8 *)(a2 - 16);
        if ( (result & 2) != 0 )
        {
          v6 = *(unsigned __int64 **)(a2 - 32);
        }
        else
        {
          result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
          v6 = (unsigned __int64 *)(a2 - 16 - result);
        }
        v3 = *v6;
      }
      if ( v3 )
      {
        v13 = 5637;
        v12[0] = 0;
        v14 = 0;
        v15 = sub_31FD260(a1, v3);
        v16 = v7;
        v8 = sub_370B390(a1 + 81, &v13);
        v9 = sub_3707F80(a1 + 79, v8);
        v10 = *(_DWORD *)(a2 + 16);
        LOWORD(v12[0]) = 5638;
        *(_DWORD *)((char *)v12 + 2) = a3;
        *(_DWORD *)((char *)&v12[1] + 2) = v9;
        v12[3] = v10;
        v11 = sub_370B620(a1 + 81, v12);
        return sub_3707F80(a1 + 79, v11);
      }
    }
  }
  return result;
}
