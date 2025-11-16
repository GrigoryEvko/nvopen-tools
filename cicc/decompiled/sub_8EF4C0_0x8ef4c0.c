// Function: sub_8EF4C0
// Address: 0x8ef4c0
//
void *__fastcall sub_8EF4C0(_DWORD *a1, char *a2, int a3)
{
  int v5; // eax
  int v6; // edx
  int v7; // r13d
  int v8; // eax
  int v9; // edx
  void *result; // rax
  int v11; // ecx
  char *v12; // rcx
  int v13; // edi
  int v14; // eax
  int v15; // edx

  v5 = sub_8EE4D0(a2, a3);
  v6 = a1[7];
  v7 = v5;
  if ( a3 == v5 )
  {
    *a1 = 6;
    v8 = v6 + 7;
  }
  else if ( a3 - v5 - v6 < 0 )
  {
    sub_8EE880(a2, a3, v6 - (a3 - v5));
    v6 = a1[7];
    v8 = v6 + 7;
  }
  else
  {
    v8 = v6 + 7;
    if ( a3 - v7 != v6 )
    {
      sub_8EE740(a2, a3, a3 - v7 - v6);
      v6 = a1[7];
      v8 = v6 + 7;
      if ( v6 < a3 )
      {
        v11 = a1[7];
        if ( v6 < 0 )
          v11 = v6 + 7;
        v12 = &a2[v11 >> 3];
        v13 = (unsigned __int8)*v12;
        if ( _bittest(&v13, v6 % 8) )
        {
          if ( (v6 & 7) != 0 )
          {
            *v12 = (unsigned __int8)*v12 >> 1;
          }
          else
          {
            v14 = v6 + 6;
            v15 = v6 - 1;
            if ( v15 >= 0 )
              v14 = v15;
            a2[v14 >> 3] = 0x80;
          }
          v6 = a1[7];
          --v7;
          v8 = v6 + 7;
        }
      }
    }
  }
  v9 = v6 + 14;
  if ( v8 >= 0 )
    v9 = v8;
  result = memcpy(a1 + 3, a2, v9 >> 3);
  a1[2] -= v7;
  return result;
}
