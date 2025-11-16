// Function: sub_39F8BA0
// Address: 0x39f8ba0
//
char *__fastcall sub_39F8BA0(char a1, char *a2, char *a3, unsigned __int64 *a4)
{
  unsigned __int64 v7; // rdx
  char *result; // rax
  unsigned __int64 *v9; // rax
  int v10; // ecx
  char v11; // r11
  unsigned __int64 v12; // r10
  unsigned int v13; // ecx
  char v14; // r11
  unsigned __int64 v15; // r10

  if ( a1 == 80 )
  {
    v9 = (unsigned __int64 *)((unsigned __int64)(a3 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *a4 = *v9;
    return (char *)(v9 + 1);
  }
  else
  {
    switch ( a1 & 0xF )
    {
      case 0:
      case 4:
      case 0xC:
        v7 = *(_QWORD *)a3;
        result = a3 + 8;
        goto LABEL_4;
      case 1:
        result = a3;
        v7 = 0;
        v10 = 0;
        do
        {
          v11 = *result++;
          v12 = (unsigned __int64)(v11 & 0x7F) << v10;
          v10 += 7;
          v7 |= v12;
        }
        while ( v11 < 0 );
        goto LABEL_4;
      case 2:
        v7 = *(unsigned __int16 *)a3;
        result = a3 + 2;
        goto LABEL_4;
      case 3:
        v7 = *(unsigned int *)a3;
        result = a3 + 4;
        goto LABEL_4;
      case 9:
        result = a3;
        v7 = 0;
        v13 = 0;
        do
        {
          v14 = *result++;
          v15 = (unsigned __int64)(v14 & 0x7F) << v13;
          v13 += 7;
          v7 |= v15;
        }
        while ( v14 < 0 );
        if ( v13 > 0x3F || (v14 & 0x40) == 0 )
          goto LABEL_4;
        v7 |= -1LL << v13;
        goto LABEL_5;
      case 0xA:
        v7 = *(__int16 *)a3;
        result = a3 + 2;
        goto LABEL_4;
      case 0xB:
        v7 = *(int *)a3;
        result = a3 + 4;
LABEL_4:
        if ( !v7 )
          goto LABEL_8;
LABEL_5:
        if ( (a1 & 0x70) == 0x10 )
          a2 = a3;
        v7 += (unsigned __int64)a2;
        if ( a1 < 0 )
          *a4 = *(_QWORD *)v7;
        else
LABEL_8:
          *a4 = v7;
        break;
      default:
        abort();
    }
  }
  return result;
}
