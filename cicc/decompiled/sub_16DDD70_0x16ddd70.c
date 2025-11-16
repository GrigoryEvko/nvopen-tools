// Function: sub_16DDD70
// Address: 0x16ddd70
//
void __fastcall sub_16DDD70(char *a1, __int64 a2, _DWORD *a3, _DWORD *a4, _DWORD *a5)
{
  char *v5; // r10
  int *v6; // rbx
  int i; // edx
  int v8; // eax
  _QWORD v9[3]; // [rsp+0h] [rbp-28h] BYREF
  char v10; // [rsp+18h] [rbp-10h] BYREF

  v5 = (char *)v9;
  *a5 = 0;
  v9[0] = a3;
  *a4 = 0;
  v9[1] = a4;
  *a3 = 0;
  v9[2] = a5;
  do
  {
    if ( !a2 || (unsigned __int8)(*a1 - 48) > 9u )
      break;
    v6 = *(int **)v5;
    for ( i = *a1 - 48; ; i = v8 + 10 * i - 48 )
    {
      if ( a2 == 1 )
      {
        *v6 = i;
        return;
      }
      v8 = a1[1];
      if ( (unsigned __int8)(a1[1] - 48) > 9u )
        break;
      ++a1;
      --a2;
    }
    *v6 = i;
    if ( a1[1] == 46 )
    {
      a2 -= 2;
      a1 += 2;
    }
    else
    {
      --a2;
      ++a1;
    }
    v5 += 8;
  }
  while ( &v10 != v5 );
}
