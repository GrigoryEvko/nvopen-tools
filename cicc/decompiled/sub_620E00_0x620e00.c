// Function: sub_620E00
// Address: 0x620e00
//
__int64 __fastcall sub_620E00(_WORD *a1, int a2, __int64 *a3, int *a4)
{
  int v4; // eax
  __int64 v8; // rcx
  int v9; // r11d
  unsigned __int16 *v10; // rdi
  int v11; // ebx
  int v12; // edx
  int v13; // esi
  __int64 result; // rax

  v8 = (unsigned __int16)*a1;
  v9 = *a1 >> 15;
  LOBYTE(v4) = a2 != 0;
  v10 = a1 + 1;
  v11 = 0;
  v12 = 0;
  v13 = -(v9 & v4);
  result = 0;
  while ( v12 <= 63 )
  {
    if ( (_WORD)v13 != (_WORD)v8 )
      v11 = 1;
    v12 += 16;
LABEL_2:
    v8 = *v10++;
  }
  v12 += 16;
  result = v8 + (result << 16);
  if ( v12 != 128 )
    goto LABEL_2;
  if ( a2 && (_BYTE)v9 != result < 0 )
    v11 = 1;
  *a3 = result;
  *a4 = v11;
  return result;
}
