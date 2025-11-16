// Function: sub_2AAAD60
// Address: 0x2aaad60
//
bool __fastcall sub_2AAAD60(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r13
  __int64 v4; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rdx

  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 == a1 + 48 )
    return 1;
  while ( 1 )
  {
    while ( 1 )
    {
      v4 = v3 - 24;
      if ( !v3 )
        v4 = 0;
      if ( *(_BYTE *)(a2 + 540) )
      {
        v5 = *(_QWORD **)(a2 + 520);
        v6 = &v5[*(unsigned int *)(a2 + 532)];
        if ( v5 != v6 )
        {
          while ( v4 != *v5 )
          {
            if ( v6 == ++v5 )
              goto LABEL_12;
          }
          goto LABEL_9;
        }
      }
      else if ( sub_C8CA60(a2 + 512, v4) )
      {
        goto LABEL_9;
      }
LABEL_12:
      if ( !*(_BYTE *)(a2 + 700) )
        break;
      v8 = *(_QWORD **)(a2 + 680);
      v9 = &v8[*(unsigned int *)(a2 + 692)];
      if ( v8 == v9 )
        goto LABEL_19;
      while ( v4 != *v8 )
      {
        if ( v9 == ++v8 )
          goto LABEL_19;
      }
LABEL_9:
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        return 1;
    }
    if ( sub_C8CA60(a2 + 672, v4) )
      goto LABEL_9;
LABEL_19:
    if ( *(_BYTE *)v4 != 31 || (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) == 3 )
      return v2 == v3;
    v3 = *(_QWORD *)(v3 + 8);
    if ( v2 == v3 )
      return 1;
  }
}
