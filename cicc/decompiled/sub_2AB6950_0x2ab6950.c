// Function: sub_2AB6950
// Address: 0x2ab6950
//
__int64 __fastcall sub_2AB6950(__int64 a1, __int64 a2)
{
  char *v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r13
  int v6; // r14d
  __int64 v8; // r13
  char *v9; // r15
  __int64 v10; // rax

  v2 = (char *)a2;
  LODWORD(v5) = sub_31A5290(*(_QWORD *)(a1 + 440), a2);
  if ( (_BYTE)v5 )
  {
    if ( *(_BYTE *)a2 > 0x1Cu )
    {
      v6 = sub_B19060(*(_QWORD *)(a1 + 416) + 56LL, *(_QWORD *)(a2 + 40), v3, v4);
      if ( (_BYTE)v6 )
      {
        LODWORD(v5) = sub_2AB37C0(a1, (unsigned __int8 *)a2);
        if ( (_BYTE)v5 )
        {
          LODWORD(v5) = 0;
          return (unsigned int)v5;
        }
        if ( *(_BYTE *)a2 != 84 || *(_QWORD *)(a2 + 40) != **(_QWORD **)(*(_QWORD *)(a1 + 416) + 32LL) )
        {
          v8 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          {
            v9 = *(char **)(a2 - 8);
            v2 = &v9[v8];
          }
          else
          {
            v9 = (char *)(a2 - v8);
          }
          v10 = v8 >> 5;
          v5 = v8 >> 7;
          if ( v5 )
          {
            v5 = (__int64)&v9[128 * v5];
            while ( (unsigned __int8)sub_2AB6950(a1, *(_QWORD *)v9) )
            {
              if ( !(unsigned __int8)sub_2AB6950(a1, *((_QWORD *)v9 + 4)) )
              {
                LOBYTE(v5) = v2 == v9 + 32;
                return (unsigned int)v5;
              }
              if ( !(unsigned __int8)sub_2AB6950(a1, *((_QWORD *)v9 + 8)) )
              {
                LOBYTE(v5) = v2 == v9 + 64;
                return (unsigned int)v5;
              }
              if ( !(unsigned __int8)sub_2AB6950(a1, *((_QWORD *)v9 + 12)) )
              {
                LOBYTE(v5) = v2 == v9 + 96;
                return (unsigned int)v5;
              }
              v9 += 128;
              if ( (char *)v5 == v9 )
              {
                v10 = (v2 - v9) >> 5;
                goto LABEL_21;
              }
            }
            goto LABEL_17;
          }
LABEL_21:
          if ( v10 != 2 )
          {
            if ( v10 != 3 )
            {
              LODWORD(v5) = v6;
              if ( v10 != 1 )
                return (unsigned int)v5;
              goto LABEL_24;
            }
            if ( !(unsigned __int8)sub_2AB6950(a1, *(_QWORD *)v9) )
            {
LABEL_17:
              LOBYTE(v5) = v2 == v9;
              return (unsigned int)v5;
            }
            v9 += 32;
          }
          if ( (unsigned __int8)sub_2AB6950(a1, *(_QWORD *)v9) )
          {
            v9 += 32;
LABEL_24:
            LODWORD(v5) = sub_2AB6950(a1, *(_QWORD *)v9);
            if ( (_BYTE)v5 )
              return (unsigned int)v5;
            goto LABEL_17;
          }
          goto LABEL_17;
        }
      }
    }
  }
  return (unsigned int)v5;
}
