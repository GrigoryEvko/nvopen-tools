// Function: sub_AA5BA0
// Address: 0xaa5ba0
//
__int64 __fastcall sub_AA5BA0(__int64 a1)
{
  __int64 v1; // r13
  __int64 result; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al
  __int64 v7; // rdx

  v1 = a1 + 48;
  result = sub_AA4FF0(a1);
  if ( result != a1 + 48 )
  {
    v3 = result;
    if ( !result )
      BUG();
    v4 = (unsigned int)*(unsigned __int8 *)(result - 24) - 39;
    if ( (unsigned int)v4 <= 0x38 && (v5 = 0x100060000000001LL, _bittest64(&v5, v4)) )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( sub_AA5B70(a1) && v3 != v1 )
      {
        do
        {
LABEL_6:
          if ( !v3 )
            BUG();
          v6 = *(_BYTE *)(v3 - 24);
          if ( v6 == 60 )
          {
            if ( !(unsigned __int8)sub_B4D040(v3 - 24) )
              return v3;
          }
          else
          {
            if ( v6 != 85 )
              return v3;
            v7 = *(_QWORD *)(v3 - 56);
            if ( !v7 )
              return v3;
            if ( (*(_BYTE *)v7
               || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v3 + 56)
               || (*(_BYTE *)(v7 + 33) & 0x20) == 0
               || (unsigned int)(*(_DWORD *)(v7 + 36) - 68) > 3)
              && (*(_BYTE *)v7
               || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v3 + 56)
               || (*(_BYTE *)(v7 + 33) & 0x20) == 0
               || *(_DWORD *)(v7 + 36) != 291) )
            {
              return v3;
            }
          }
          v3 = *(_QWORD *)(v3 + 8);
        }
        while ( v1 != v3 );
      }
    }
    else if ( sub_AA5B70(a1) )
    {
      goto LABEL_6;
    }
    return v3;
  }
  return result;
}
