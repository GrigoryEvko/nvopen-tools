// Function: sub_85F9C0
// Address: 0x85f9c0
//
__int64 __fastcall sub_85F9C0(__int64 *a1)
{
  __int64 v2; // rdi
  char v3; // al
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rbx
  char v9; // al
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax

  v2 = *a1;
  v3 = *(_BYTE *)(v2 + 80);
  if ( v3 == 3 )
  {
    if ( !*(_BYTE *)(v2 + 104) )
      return 1;
    v12 = *(_QWORD *)(v2 + 88);
    if ( (*(_BYTE *)(v12 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v12 + 168) + 168LL) )
      return 1;
    v13 = sub_880FE0(v2);
    *a1 = v13;
    v3 = *(_BYTE *)(v13 + 80);
  }
  if ( v3 != 19 )
    return 1;
  if ( unk_4F04C28 > 0 || unk_4F04C48 != -1 )
  {
    v5 = dword_4F04C64;
    while ( 1 )
    {
      v8 = qword_4F04C68[0] + 776 * v5;
      if ( !v8 )
        return 0;
      v9 = *(_BYTE *)(v8 + 4);
      if ( v9 == 9 )
      {
        if ( *(char *)(*(_QWORD *)(*a1 + 88) + 160LL) < 0 )
        {
          v10 = *(__int64 **)(v8 + 208);
          if ( v10 )
          {
            v6 = *v10;
            v11 = *(_QWORD *)(*(_QWORD *)(*v10 + 96) + 72LL);
            if ( v11 )
            {
              if ( *a1 == sub_892920(v11) )
              {
LABEL_16:
                *a1 = v6;
                return 1;
              }
            }
          }
        }
        if ( (*(_BYTE *)(v8 + 7) & 0x20) == 0 )
          return 0;
      }
      else if ( (unsigned __int8)(v9 - 6) <= 1u )
      {
        v6 = **(_QWORD **)(v8 + 208);
        v7 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 72LL);
        if ( v7 )
        {
          if ( *a1 == sub_892920(v7) )
            goto LABEL_16;
        }
        else if ( !*a1 )
        {
          goto LABEL_16;
        }
      }
      v5 = *(int *)(v8 + 552);
      if ( (_DWORD)v5 == -1 )
        return 0;
    }
  }
  return 0;
}
