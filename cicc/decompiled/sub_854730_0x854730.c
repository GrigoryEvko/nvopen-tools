// Function: sub_854730
// Address: 0x854730
//
__int64 sub_854730()
{
  __int64 result; // rax
  __int64 *v1; // r14
  __int64 *v2; // rbx
  __int64 v3; // r12

  result = (unsigned int)dword_4F04D80;
  if ( !dword_4F04D80 )
    result = (__int64)sub_853C60(3);
  v1 = (__int64 *)qword_4D03E88;
  qword_4D03E88 = 0;
  if ( v1 )
  {
    v2 = v1;
    do
    {
      while ( 1 )
      {
        v3 = v2[1];
        if ( *(_DWORD *)(v3 + 12) == 3 )
        {
          result = *((unsigned __int8 *)v2 + 72);
          if ( (result & 8) == 0 )
          {
            *((_BYTE *)v2 + 72) = result | 8;
            if ( (*(_BYTE *)(v3 + 17) & 8) != 0 )
              sub_8543B0(v2, 0, 0);
            result = off_4A51EC0[*(unsigned __int8 *)(v3 + 16)];
            if ( result )
              break;
          }
        }
        v2 = (__int64 *)*v2;
        if ( !v2 )
          goto LABEL_12;
      }
      result = ((__int64 (__fastcall *)(__int64 *))result)(v2);
      v2 = (__int64 *)*v2;
    }
    while ( v2 );
  }
LABEL_12:
  qword_4D03E88 = v1;
  return result;
}
