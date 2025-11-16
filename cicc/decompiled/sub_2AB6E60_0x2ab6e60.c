// Function: sub_2AB6E60
// Address: 0x2ab6e60
//
__int64 __fastcall sub_2AB6E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // rdi
  int v6; // r11d
  unsigned int i; // eax
  _QWORD *v8; // rdx
  unsigned int v9; // eax

  v3 = *(_QWORD *)(a1 + 72);
  v4 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v4 )
  {
    v6 = 1;
    for ( i = (v4 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v4 - 1) & v9 )
    {
      v8 = (_QWORD *)(v3 + 24LL * i);
      if ( a2 == *v8 && a3 == v8[1] )
        break;
      if ( *v8 == -4096 && v8[1] == -4096 )
        goto LABEL_7;
      v9 = v6 + i;
      ++v6;
    }
  }
  else
  {
LABEL_7:
    v8 = (_QWORD *)(v3 + 24 * v4);
  }
  return v8[2];
}
