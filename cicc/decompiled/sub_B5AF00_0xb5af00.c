// Function: sub_B5AF00
// Address: 0xb5af00
//
char __fastcall sub_B5AF00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *v3; // r12
  char result; // al
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // r13
  unsigned int v8; // r14d
  __int64 v9; // [rsp+8h] [rbp-28h]

  v9 = sub_B5A2C0(a1);
  v2 = sub_B5A450(a1);
  if ( !v2 )
    return 1;
  v3 = (unsigned __int8 *)v2;
  result = BYTE4(v9);
  v5 = *v3;
  if ( !BYTE4(v9) )
  {
    if ( (_BYTE)v5 != 17 )
      return result;
    v6 = (_QWORD *)*((_QWORD *)v3 + 3);
    if ( *((_DWORD *)v3 + 8) > 0x40u )
      v6 = (_QWORD *)*v6;
    return (unsigned int)v9 <= (unsigned __int64)v6;
  }
  if ( (_BYTE)v5 == 46 )
  {
    if ( (unsigned __int8)sub_B58A70(*((unsigned __int8 **)v3 - 8), a2, v5) )
    {
      v7 = *((_QWORD *)v3 - 4);
      if ( *(_BYTE *)v7 == 17 )
      {
        v8 = *(_DWORD *)(v7 + 32);
        if ( v8 <= 0x40 )
        {
          v6 = *(_QWORD **)(v7 + 24);
          return (unsigned int)v9 <= (unsigned __int64)v6;
        }
        if ( v8 - (unsigned int)sub_C444A0(v7 + 24) <= 0x40 )
        {
          v6 = **(_QWORD ***)(v7 + 24);
          return (unsigned int)v9 <= (unsigned __int64)v6;
        }
      }
    }
  }
  result = 0;
  if ( (_DWORD)v9 == 1 )
    return sub_B58A70(v3, a2, v5);
  return result;
}
