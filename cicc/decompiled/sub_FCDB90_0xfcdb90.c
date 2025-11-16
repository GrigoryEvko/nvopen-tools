// Function: sub_FCDB90
// Address: 0xfcdb90
//
bool __fastcall sub_FCDB90(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned __int64 v7; // r13

  if ( *(_BYTE *)a2 != 22 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 && (unsigned __int8)sub_B2D670(a2, 81) )
    return 0;
  if ( (_BYTE)qword_4F8D5E8 )
  {
    v2 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL);
    if ( (unsigned __int8)v2 <= 3u
      || (_BYTE)v2 == 5
      || (unsigned __int8)v2 <= 0x14u && (v3 = 1463376, _bittest64(&v3, v2)) )
    {
      if ( *(_BYTE *)(a1 + 204) )
      {
        v4 = *(_QWORD **)(a1 + 184);
        v5 = &v4[*(unsigned int *)(a1 + 196)];
        if ( v4 != v5 )
        {
          while ( a2 != *v4 )
          {
            if ( v5 == ++v4 )
              goto LABEL_15;
          }
          return 0;
        }
      }
      else if ( sub_C8CA60(a1 + 176, a2) )
      {
        return 0;
      }
    }
  }
LABEL_15:
  v7 = (unsigned __int64)(int)qword_4F8D888 >> 2;
  return (int)v7 >= (int)sub_FCD870(a2, *(_QWORD *)(*(_QWORD *)a1 + 40LL) + 312LL);
}
