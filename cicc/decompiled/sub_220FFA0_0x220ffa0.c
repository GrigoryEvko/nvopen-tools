// Function: sub_220FFA0
// Address: 0x220ffa0
//
__int64 __fastcall sub_220FFA0(unsigned __int16 **a1, __int64 a2, unsigned __int64 a3, int a4)
{
  unsigned __int64 v5; // r10
  int v6; // r11d
  unsigned __int16 *v8; // rax
  unsigned __int64 v9; // rcx
  int v10; // eax
  unsigned int v11; // esi
  __int64 v12; // r8
  unsigned __int16 **v13; // r9
  unsigned __int16 *v14; // rdx

  v5 = a3;
  v6 = a4;
  v8 = *a1;
  v9 = (char *)a1[1] - (char *)*a1;
  if ( a1[1] == *a1 )
    return 0;
  while ( 1 )
  {
    v11 = *v8;
    if ( v11 - 55296 <= 0x3FF )
    {
      if ( v6 == 1 )
        return 2;
      if ( v9 <= 2 )
        return 0;
      v10 = v8[1];
      if ( (unsigned int)(v10 - 56320) > 0x3FF )
        return 2;
      v11 = v10 + (v11 << 10) - 56613888;
      if ( v11 > v5 )
        return 2;
    }
    else if ( v11 - 56320 <= 0x3FF || *v8 > v5 )
    {
      return 2;
    }
    if ( !(unsigned __int8)sub_220FDA0(a2, v11) )
      break;
    v14 = v13[1];
    v8 = &(*v13)[v12];
    *v13 = v8;
    v9 = (char *)v14 - (char *)v8;
    if ( v14 == v8 )
      return 0;
  }
  return 1;
}
