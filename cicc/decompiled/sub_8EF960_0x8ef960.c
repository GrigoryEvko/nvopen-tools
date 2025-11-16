// Function: sub_8EF960
// Address: 0x8ef960
//
void *__fastcall sub_8EF960(_DWORD *a1, _DWORD *a2)
{
  void *result; // rax
  bool v3; // zf
  int v6; // r14d
  int v7; // r15d
  int v8; // r13d
  int v9; // r11d
  __int64 v10; // rdi
  __int128 *v11; // rsi
  int v12; // r11d
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rcx
  int v16; // eax
  int v17; // eax
  int v18; // edx
  __int128 v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+20h] [rbp-40h]
  int v21; // [rsp+28h] [rbp-38h]
  __int16 v22; // [rsp+2Ch] [rbp-34h]
  char v23; // [rsp+2Eh] [rbp-32h]

  result = 0;
  v3 = *a1 == 6;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v19 = 0;
  if ( !v3 )
  {
    if ( *a2 == 6 )
    {
      *a1 = 6;
    }
    else
    {
      v6 = a1[7];
      if ( v6 > 0 )
      {
        v7 = a2[7];
        v8 = (v6 + 7) >> 3;
        v9 = v7 + 14;
        if ( v7 + 7 >= 0 )
          v9 = v7 + 7;
        v10 = 0;
        v11 = &v19;
        v12 = v9 >> 3;
        do
        {
          while ( 1 )
          {
            v13 = *((unsigned __int8 *)a1 + v10 + 12);
            v14 = v10;
            if ( (_BYTE)v13 )
              break;
            ++v10;
            v11 = (__int128 *)((char *)v11 + 1);
            if ( v8 <= (int)v10 )
              goto LABEL_16;
          }
          if ( v7 <= 0 )
          {
            LOBYTE(v16) = 0;
          }
          else
          {
            v15 = 0;
            v16 = 0;
            do
            {
              v17 = *((unsigned __int8 *)v11 + v15) + v13 * *((unsigned __int8 *)a2 + v15 + 12) + v16;
              *((_BYTE *)v11 + v15++) = v17;
              v16 = v17 >> 8;
            }
            while ( v12 > (int)v15 );
            v18 = 1;
            if ( v12 > 0 )
              v18 = v12;
            v14 = v18 + v10;
          }
          ++v10;
          v11 = (__int128 *)((char *)v11 + 1);
          *((_BYTE *)&v19 + v14) = v16;
        }
        while ( v8 > (int)v10 );
      }
LABEL_16:
      a1[2] += a2[2];
      return sub_8EF4C0(a1, (char *)&v19, v6 + a2[7]);
    }
  }
  return result;
}
