// Function: sub_1694D40
// Address: 0x1694d40
//
__int64 *__fastcall sub_1694D40(__int64 *a1, unsigned int *a2)
{
  unsigned int v3; // r10d
  signed __int64 v4; // r11
  _DWORD *v6; // rdi
  int v7; // r8d
  int v8; // esi
  unsigned __int8 *v9; // rax
  int v10; // edx
  int v11; // ecx
  __int64 v12; // rdx

  v3 = a2[1];
  if ( v3 <= 2 )
  {
    v4 = *a2;
    if ( (v4 & 7) == 0 )
    {
      v6 = a2 + 2;
      if ( !v3 )
      {
LABEL_12:
        *a1 = 1;
        return a1;
      }
      v7 = 0;
      while ( *v6 <= 1u )
      {
        v8 = v6[1];
        if ( v8 )
        {
          v9 = (unsigned __int8 *)(v6 + 2);
          v10 = 0;
          do
          {
            v11 = *v9++;
            v10 += v11;
          }
          while ( (unsigned __int8 *)((char *)v6 + (unsigned int)(v8 - 1) + 9) != v9 );
          v12 = ((v8 + 15) & 0xFFFFFFF8) + 16 * v10;
        }
        else
        {
          v12 = 8;
        }
        v6 = (_DWORD *)((char *)v6 + v12);
        if ( (char *)v6 - (char *)a2 > v4 )
          break;
        if ( v3 == ++v7 )
          goto LABEL_12;
      }
    }
  }
  sub_1693CB0(a1, 9);
  return a1;
}
