// Function: sub_F57050
// Address: 0xf57050
//
void __fastcall sub_F57050(unsigned __int8 *a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r14
  unsigned int *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp-40h] [rbp-40h] BYREF
  __int64 v14; // [rsp-38h] [rbp-38h]
  __int64 v15; // [rsp-30h] [rbp-30h]

  v2 = *a2;
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
    v3 = *a1;
    if ( (unsigned __int8)v2 <= 0x36u && (v4 = 0x40540000000000LL, _bittest64(&v4, v2)) && v3 == 93 )
    {
      if ( *((_DWORD *)a1 + 20) == 1 && !**((_DWORD **)a1 + 9) )
      {
        v5 = *((_QWORD *)a1 - 4);
        if ( *(_BYTE *)v5 == 85 )
        {
          v6 = *(_QWORD *)(v5 - 32);
          if ( v6 )
          {
            if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *(_QWORD *)(v5 + 80) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
            {
              v7 = *(_DWORD *)(v6 + 36);
              if ( v7 != 312 )
              {
                switch ( v7 )
                {
                  case 333:
                  case 339:
                  case 360:
                  case 369:
                  case 372:
                    break;
                  default:
                    goto LABEL_16;
                }
              }
              sub_B44F30(a2);
              goto LABEL_17;
            }
          }
        }
      }
    }
    else if ( v3 == 61 )
    {
LABEL_15:
      sub_F57030(a2, (__int64)a1, 0);
      return;
    }
LABEL_16:
    sub_B45560(a2, (unsigned __int64)a1);
LABEL_17:
    if ( (unsigned __int8)(*a2 - 34) <= 0x33u )
    {
      v8 = 0x8000000000041LL;
      if ( _bittest64(&v8, (unsigned int)*a2 - 34) )
      {
        if ( (unsigned __int8)(*a1 - 34) <= 0x33u && _bittest64(&v8, (unsigned int)*a1 - 34) && a2 != a1 )
        {
          v9 = *((_QWORD *)a1 + 9);
          v13 = *((_QWORD *)a2 + 9);
          v10 = (unsigned int *)sub_BD5C60((__int64)a2);
          v11 = sub_A7AD50(&v13, v10, v9);
          v15 = v12;
          v14 = v11;
          if ( (_BYTE)v12 )
            *((_QWORD *)a2 + 9) = v14;
        }
      }
    }
    goto LABEL_15;
  }
}
