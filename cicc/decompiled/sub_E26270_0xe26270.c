// Function: sub_E26270
// Address: 0xe26270
//
__int64 __fastcall sub_E26270(__int64 a1, __int64 *a2)
{
  char *v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  _WORD *v5; // r14
  int v6; // ecx
  unsigned __int64 v8; // rbx
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // rcx
  char v12; // al

  v4 = *a2;
  if ( !*a2 )
    return sub_E22500(a1, a2, 1);
  v5 = (_WORD *)a2[1];
  v6 = *(char *)v5;
  if ( (unsigned int)(v6 - 48) > 9 )
  {
    if ( v4 != 1 )
    {
      if ( *v5 == 9279 )
        return sub_E24FC0(a1, (size_t *)a2, 1);
      if ( *v5 == 16703 )
        return sub_E22980(a1, (size_t *)a2);
      if ( (_BYTE)v6 == 63 )
      {
        v8 = v4 - 1;
        v9 = memchr((char *)v5 + 1, 63, v4 - 1);
        if ( v9 )
        {
          v10 = v9 - ((char *)v5 + 1);
          if ( v10 != -1 )
          {
            if ( v10 <= v8 )
            {
              if ( !v10 )
                return sub_E22500(a1, a2, 1);
              v8 = v10;
            }
            if ( v8 == 1 )
            {
              v12 = *((_BYTE *)v5 + 1);
              if ( v12 == 64 || (unsigned __int8)(v12 - 48) <= 9u )
                return sub_E25E60(a1, (unsigned __int64 *)a2);
            }
            else if ( *((_BYTE *)v5 + v8) == 64 && (unsigned __int8)(*((_BYTE *)v5 + 1) - 66) <= 0xEu )
            {
              if ( v8 != 2 )
              {
                v11 = v5 + 1;
                while ( (unsigned __int8)(*v11 - 65) <= 0xFu )
                {
                  if ( ++v11 == (char *)v5 + v8 )
                    return sub_E25E60(a1, (unsigned __int64 *)a2);
                }
                return sub_E22500(a1, a2, 1);
              }
              return sub_E25E60(a1, (unsigned __int64 *)a2);
            }
          }
        }
      }
    }
    return sub_E22500(a1, a2, 1);
  }
  v2 = (char *)a2[1];
  v3 = *v2 - 48;
  if ( *(_QWORD *)(a1 + 192) <= v3 )
    JUMPOUT(0xE21CB0);
  --*a2;
  a2[1] = (__int64)(v2 + 1);
  return *(_QWORD *)(a1 + 8 * v3 + 112);
}
