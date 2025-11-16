// Function: sub_5CB110
// Address: 0x5cb110
//
__int64 __fastcall sub_5CB110(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // r13
  _QWORD *v5; // rbx
  __int64 v7; // [rsp+8h] [rbp-38h] BYREF
  __int64 v8[5]; // [rsp+18h] [rbp-28h] BYREF

  v7 = a2;
  v3 = sub_5C7B50(a1, (__int64)&v7, a3);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 168);
    if ( (*(_BYTE *)(v4 + 16) & 1) != 0 )
    {
      v5 = *(_QWORD **)(a1 + 32);
      if ( v5 )
      {
        do
        {
          if ( (unsigned int)sub_5CACA0((__int64)v5, a1, 0, 2147483646, v8) )
            *(_DWORD *)(v4 + 36) = LODWORD(v8[0]) + 1;
          v5 = (_QWORD *)*v5;
        }
        while ( v5 );
      }
      else
      {
        *(_DWORD *)(v4 + 36) = 1;
      }
    }
    else
    {
      sub_6851C0(1545, a1 + 56);
    }
  }
  return v7;
}
