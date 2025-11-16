// Function: sub_24ABC70
// Address: 0x24abc70
//
__int64 __fastcall sub_24ABC70(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // rcx
  _BYTE *v7; // rdx
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( byte_4FEC008 )
  {
    v2 = sub_ED2C10(a1, 1);
    if ( (_BYTE)v2 )
    {
      v8[0] = *(_QWORD *)(a1 + 48);
      v4 = (_QWORD *)sub_24ABBE0(a2, v8);
      v6 = v5;
      if ( v4 == v5 )
        return v2;
      while ( 1 )
      {
        v7 = (_BYTE *)v4[2];
        if ( *v7 || (_BYTE *)a1 != v7 )
          break;
        v4 = (_QWORD *)*v4;
        if ( v6 == v4 )
          return v2;
      }
    }
  }
  return 0;
}
