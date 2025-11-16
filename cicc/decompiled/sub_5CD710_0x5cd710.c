// Function: sub_5CD710
// Address: 0x5cd710
//
__int64 __fastcall sub_5CD710(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // [rsp+8h] [rbp-28h] BYREF
  __int64 v11; // [rsp+10h] [rbp-20h] BYREF
  __int64 v12[3]; // [rsp+18h] [rbp-18h] BYREF

  v10 = a2;
  if ( (unsigned __int8)a3 <= 0xBu && ((1LL << a3) & 0x9C8) != 0 && (v4 = sub_5C7B50(a1, (__int64)&v10, a3)) != 0 )
  {
    v5 = *(_QWORD *)(v4 + 168);
    v12[0] = 0;
    v6 = *(_QWORD **)v5;
    v7 = *(_QWORD *)(v5 + 40) != 0;
    if ( *(_QWORD *)v5 )
    {
      do
      {
        v6 = (_QWORD *)*v6;
        ++v7;
      }
      while ( v6 );
    }
    if ( !(unsigned int)sub_5CACA0(*(_QWORD *)(a1 + 32), a1, 1, v7, &v11)
      || (v8 = **(_QWORD **)(a1 + 32)) != 0 && !(unsigned int)sub_5CACA0(v8, a1, 1, v7, v12) )
    {
      result = v10;
      *(_BYTE *)(a1 + 8) = 0;
      return result;
    }
  }
  else
  {
    sub_5CCAE0(5u, a1);
  }
  return v10;
}
