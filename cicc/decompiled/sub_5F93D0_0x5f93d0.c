// Function: sub_5F93D0
// Address: 0x5f93d0
//
__int64 __fastcall sub_5F93D0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  char v4; // dl
  __int64 v5; // r13
  __int64 v6; // r14

  result = (__int64)&dword_4D048B8;
  if ( dword_4D048B8 )
  {
    result = (__int64)&dword_4D048B0;
    if ( dword_4D048B0 )
    {
      if ( (*(_BYTE *)(a1 + 195) & 8) == 0 )
      {
        result = *a2;
        if ( *(_BYTE *)(*a2 + 140) == 7 )
        {
          v3 = *(_QWORD *)(result + 168);
          if ( !*(_QWORD *)(v3 + 56) )
          {
            result = *(_QWORD *)(a1 + 152);
            if ( *(_BYTE *)(result + 140) == 7 )
            {
              v4 = *(_BYTE *)(a1 + 174);
              if ( v4 == 5 )
              {
                result = (unsigned int)*(unsigned __int8 *)(a1 + 176) - 2;
                if ( ((*(_BYTE *)(a1 + 176) - 2) & 0xFD) == 0 )
                  return sub_5F1D90(v3);
              }
              else if ( v4 == 2 )
              {
                v5 = *(_QWORD *)(result + 168);
                v6 = *(_QWORD *)(v5 + 56);
                *(_QWORD *)(v5 + 56) = 0;
                result = (__int64)sub_5F8DB0(a1, 0);
                if ( a2 != (__int64 *)(a1 + 152) )
                {
                  result = *(_QWORD *)(v5 + 56);
                  *(_QWORD *)(v3 + 56) = result;
                  *(_QWORD *)(v5 + 56) = v6;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
