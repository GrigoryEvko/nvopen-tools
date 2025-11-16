// Function: sub_8D1880
// Address: 0x8d1880
//
__int64 __fastcall sub_8D1880(__int64 a1, _DWORD *a2)
{
  char v3; // al
  __int64 *v4; // rax
  char v5; // dl
  __int64 v6; // rax
  __int64 *v7[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
    return 0;
  v3 = *(_BYTE *)(a1 + 177);
  if ( (v3 & 0x10) == 0 )
  {
LABEL_4:
    if ( (v3 & 0x20) != 0 )
    {
      v4 = *(__int64 **)(*(_QWORD *)(a1 + 168) + 168LL);
      v7[0] = v4;
      if ( v4 )
      {
        v5 = *((_BYTE *)v4 + 8);
        if ( v5 != 3 )
          goto LABEL_7;
        sub_72F220(v7);
        v4 = v7[0];
        if ( v7[0] )
        {
          v5 = *((_BYTE *)v7[0] + 8);
          while ( 1 )
          {
LABEL_7:
            if ( v5 == 2 )
            {
LABEL_12:
              if ( (unsigned int)sub_89AAB0(v4[4], qword_4F60560, 0) )
                goto LABEL_13;
            }
            while ( 1 )
            {
              v4 = (__int64 *)*v7[0];
              v7[0] = v4;
              if ( !v4 )
                return 0;
              v5 = *((_BYTE *)v4 + 8);
              if ( v5 != 3 )
                break;
              sub_72F220(v7);
              v4 = v7[0];
              if ( !v7[0] )
                return 0;
              if ( *((_BYTE *)v7[0] + 8) == 2 )
                goto LABEL_12;
            }
          }
        }
      }
    }
    return 0;
  }
  v6 = sub_880FA0(a1);
  if ( v6 && (unsigned int)sub_89AAB0(*(_QWORD *)(*(_QWORD *)(v6 + 88) + 104LL), qword_4F60560, 0) )
  {
LABEL_13:
    *a2 = 1;
    return 1;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    v3 = *(_BYTE *)(a1 + 177);
    goto LABEL_4;
  }
  return 0;
}
