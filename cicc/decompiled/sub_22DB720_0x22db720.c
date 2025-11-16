// Function: sub_22DB720
// Address: 0x22db720
//
__int64 __fastcall sub_22DB720(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // eax
  char v8; // al
  __int64 v10; // rdx

  v1 = *(_QWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( !v1 )
    return 0;
  while ( 1 )
  {
    v2 = *(_QWORD *)(v1 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v2 - 30) <= 0xAu )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 0;
  }
  v3 = *(_QWORD *)(v2 + 40);
  v4 = 0;
  v5 = a1[3];
  if ( v3 )
  {
LABEL_4:
    v6 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
    v7 = *(_DWORD *)(v3 + 44) + 1;
  }
  else
  {
LABEL_16:
    v6 = 0;
    v7 = 0;
  }
  if ( v7 < *(_DWORD *)(v5 + 32) )
  {
    if ( *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v6) )
    {
      v8 = sub_22DB400(a1, v3);
      if ( v3 )
      {
        if ( v8 != 1 )
        {
          if ( !v4 )
          {
            v4 = v3;
            goto LABEL_13;
          }
          return 0;
        }
      }
    }
  }
LABEL_13:
  while ( 1 )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v4;
    v10 = *(_QWORD *)(v1 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
    {
      v3 = *(_QWORD *)(v10 + 40);
      v5 = a1[3];
      if ( v3 )
        goto LABEL_4;
      goto LABEL_16;
    }
  }
}
