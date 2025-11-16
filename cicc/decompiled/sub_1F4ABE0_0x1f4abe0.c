// Function: sub_1F4ABE0
// Address: 0x1f4abe0
//
__int64 *__fastcall sub_1F4ABE0(__int64 a1, unsigned int a2, char a3)
{
  __int64 **v4; // r10
  __int64 **v5; // rdi
  __int64 *v8; // r12
  unsigned int v9; // r13d
  unsigned int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // r11
  __int64 v13; // rcx
  char *v14; // rdx
  char v15; // al
  int v16; // eax

  v4 = *(__int64 ***)(a1 + 264);
  v5 = *(__int64 ***)(a1 + 256);
  if ( v5 != v4 )
  {
    v8 = 0;
    v9 = a2;
    v10 = a2 & 7;
    v11 = v9 >> 3;
    v12 = v4 - v5;
    while ( 1 )
    {
      v13 = **v5;
      if ( a3 == 1 )
        goto LABEL_8;
      v14 = *(char **)(*(_QWORD *)(a1 + 280)
                     + 24LL * ((_DWORD)v12 * *(_DWORD *)(a1 + 288) + (unsigned int)*(unsigned __int16 *)(v13 + 24))
                     + 16);
      v15 = *v14;
      if ( *v14 != 1 )
        break;
LABEL_14:
      if ( v4 == ++v5 )
        return v8;
    }
    while ( a3 != v15 )
    {
      v15 = *++v14;
      if ( v15 == 1 )
        goto LABEL_14;
    }
LABEL_8:
    if ( v11 < *(unsigned __int16 *)(v13 + 22) )
    {
      v16 = *(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + v11);
      if ( _bittest(&v16, v10) )
      {
        if ( v8 )
        {
          if ( *v5 != v8
            && ((*(_DWORD *)(v8[1] + 4 * ((unsigned __int64)*(unsigned __int16 *)(v13 + 24) >> 5)) >> *(_WORD *)(v13 + 24))
              & 1) != 0 )
          {
            v8 = *v5;
          }
        }
        else
        {
          v8 = *v5;
        }
      }
    }
    goto LABEL_14;
  }
  return 0;
}
