// Function: sub_373FC60
// Address: 0x373fc60
//
unsigned __int8 *__fastcall sub_373FC60(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned int v8; // ecx
  unsigned __int8 **v9; // rdx
  unsigned __int8 *v10; // r8
  int v11; // edx
  int v12; // r9d

  v2 = a2;
  if ( a2 )
  {
    v3 = *a2;
    if ( (unsigned __int8)(*a2 - 18) <= 2u )
    {
      if ( v3 == 20 )
      {
        v2 = sub_AF3520(a2);
        v3 = *v2;
      }
      if ( v3 == 19 )
        return sub_3737680((__int64)a1, v2);
      v5 = !sub_3734FE0((__int64)a1) || (unsigned __int8)sub_321F6A0(a1[26], a2) ? a1[27] + 400 : (__int64)(a1 + 84);
      v6 = *(_QWORD *)(v5 + 8);
      v7 = *(unsigned int *)(v5 + 24);
      if ( (_DWORD)v7 )
      {
        v8 = (v7 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v9 = (unsigned __int8 **)(v6 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == v2 )
        {
LABEL_11:
          if ( v9 != (unsigned __int8 **)(v6 + 16 * v7) )
            return v9[1];
        }
        else
        {
          v11 = 1;
          while ( v10 != (unsigned __int8 *)-4096LL )
          {
            v12 = v11 + 1;
            v8 = (v7 - 1) & (v11 + v8);
            v9 = (unsigned __int8 **)(v6 + 16LL * v8);
            v10 = *v9;
            if ( *v9 == v2 )
              goto LABEL_11;
            v11 = v12;
          }
        }
      }
    }
  }
  return sub_3250780(a1, (__int64)v2);
}
