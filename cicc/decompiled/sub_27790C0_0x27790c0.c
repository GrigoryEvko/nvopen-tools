// Function: sub_27790C0
// Address: 0x27790c0
//
unsigned __int64 __fastcall sub_27790C0(unsigned __int8 *a1, unsigned __int8 *a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // r13
  int v4; // edx
  unsigned int v5; // ecx
  unsigned __int8 v6; // al
  __int64 *v7; // rax
  unsigned __int8 *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned int *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // [rsp-40h] [rbp-40h] BYREF
  unsigned __int64 v16; // [rsp-38h] [rbp-38h]
  __int64 v17; // [rsp-30h] [rbp-30h]

  result = *a2;
  if ( (unsigned __int8)result > 0x1Cu )
  {
    switch ( *a2 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_28;
      case 'T':
      case 'U':
      case 'V':
        v3 = *((_QWORD *)a2 + 1);
        v4 = *(unsigned __int8 *)(v3 + 8);
        v5 = v4 - 17;
        v6 = *(_BYTE *)(v3 + 8);
        if ( (unsigned int)(v4 - 17) <= 1 )
          v6 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( v6 <= 3u || v6 == 5 || (v6 & 0xFD) == 4 )
          goto LABEL_28;
        if ( (_BYTE)v4 == 15 )
        {
          if ( (*(_BYTE *)(v3 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a2 + 1)) )
            goto LABEL_17;
          v7 = *(__int64 **)(v3 + 16);
          v3 = *v7;
          v4 = *(unsigned __int8 *)(*v7 + 8);
          v5 = v4 - 17;
        }
        else if ( (_BYTE)v4 == 16 )
        {
          do
          {
            v3 = *(_QWORD *)(v3 + 24);
            LOBYTE(v4) = *(_BYTE *)(v3 + 8);
          }
          while ( (_BYTE)v4 == 16 );
          v5 = (unsigned __int8)v4 - 17;
        }
        if ( v5 <= 1 )
          LOBYTE(v4) = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( (unsigned __int8)v4 <= 3u || (_BYTE)v4 == 5 || (v4 & 0xFD) == 4 )
          goto LABEL_28;
LABEL_17:
        if ( (unsigned __int8)sub_B44920(a2) && !(unsigned __int8)sub_98F650((__int64)a2, (__int64)a2, v8, v9, v10) )
LABEL_28:
          sub_B45560(a2, (unsigned __int64)a1);
        result = (unsigned int)*a1 - 34;
        if ( (unsigned __int8)(*a1 - 34) <= 0x33u )
        {
          v11 = 0x8000000000041LL;
          if ( _bittest64(&v11, result) )
          {
            result = *a2;
            if ( (unsigned __int8)result > 0x1Cu )
            {
              result = (unsigned int)(result - 34);
              if ( (unsigned __int8)result <= 0x33u && _bittest64(&v11, result) && a1 != a2 )
              {
                v12 = *((_QWORD *)a1 + 9);
                v15 = *((_QWORD *)a2 + 9);
                v13 = (unsigned int *)sub_BD5C60((__int64)a2);
                result = sub_A7AD50(&v15, v13, v12);
                v17 = v14;
                v16 = result;
                if ( (_BYTE)v14 )
                {
                  result = v16;
                  *((_QWORD *)a2 + 9) = v16;
                }
              }
            }
          }
        }
        break;
      default:
        goto LABEL_17;
    }
  }
  return result;
}
