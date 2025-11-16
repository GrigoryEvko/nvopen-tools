// Function: sub_AE8440
// Address: 0xae8440
//
unsigned __int64 __fastcall sub_AE8440(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int8 v3; // al
  __int64 v4; // r13
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned __int8 v11; // dl
  unsigned __int64 *v12; // r13
  unsigned __int64 *i; // r12
  unsigned __int8 v14; // dl
  __int64 v15; // rax
  char v16; // dl

  result = sub_AE7E70(a1, a2);
  if ( (_BYTE)result )
  {
    v3 = *(_BYTE *)(a2 - 16);
    v4 = a2 - 16;
    v5 = (v3 & 2) != 0 ? *(_QWORD *)(a2 - 32) : v4 - 8LL * ((v3 >> 2) & 0xF);
    sub_AE8080(a1, *(unsigned __int8 **)(v5 + 8));
    v6 = *(_BYTE *)(a2 - 16);
    v7 = (v6 & 2) != 0 ? *(_QWORD *)(a2 - 32) : v4 - 8LL * ((v6 >> 2) & 0xF);
    sub_AE8620(a1, *(_QWORD *)(v7 + 40));
    v8 = *(_BYTE *)(a2 - 16);
    v9 = (v8 & 2) != 0 ? *(_QWORD *)(a2 - 32) : v4 - 8LL * ((v8 >> 2) & 0xF);
    sub_AE8230(a1, *(unsigned __int8 **)(v9 + 32));
    result = *(unsigned __int8 *)(a2 - 16);
    if ( (result & 2) != 0 )
    {
      if ( *(_DWORD *)(a2 - 24) <= 9u )
        return result;
      v10 = *(_QWORD *)(a2 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) <= 9 )
        return result;
      v10 = v4 - 8LL * (((unsigned __int8)result >> 2) & 0xF);
    }
    result = *(_QWORD *)(v10 + 72);
    if ( result )
    {
      v11 = *(_BYTE *)(result - 16);
      if ( (v11 & 2) != 0 )
      {
        v12 = *(unsigned __int64 **)(result - 32);
        result = *(unsigned int *)(result - 24);
      }
      else
      {
        v12 = (unsigned __int64 *)(result - 16 - 8LL * ((v11 >> 2) & 0xF));
        result = (*(_WORD *)(result - 16) >> 6) & 0xF;
      }
      for ( i = &v12[result]; i != v12; ++v12 )
      {
        result = *v12;
        v16 = *(_BYTE *)*v12;
        if ( v16 == 23 )
        {
          v14 = *(_BYTE *)(result - 16);
          if ( (v14 & 2) == 0 )
            goto LABEL_24;
        }
        else
        {
          if ( v16 != 24 )
            continue;
          v14 = *(_BYTE *)(result - 16);
          if ( (v14 & 2) == 0 )
          {
LABEL_24:
            v15 = result - 16 - 8LL * ((v14 >> 2) & 0xF);
            goto LABEL_19;
          }
        }
        v15 = *(_QWORD *)(result - 32);
LABEL_19:
        result = sub_AE8230(a1, *(unsigned __int8 **)(v15 + 8));
      }
    }
  }
  return result;
}
