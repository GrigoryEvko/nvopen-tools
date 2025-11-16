// Function: sub_5E7470
// Address: 0x5e7470
//
__int64 __fastcall sub_5E7470(_QWORD *a1, _DWORD *a2)
{
  _QWORD *v2; // r13
  __int64 i; // rbx
  __int64 v4; // r13
  _QWORD *v5; // rbx
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 j; // rbx
  char v10; // dl
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1;
  if ( (*((_BYTE *)a1 + 89) & 4) == 0 )
    ++*a2;
  for ( i = a1[20]; i; i = *(_QWORD *)(i + 112) )
  {
    a1 = *(_QWORD **)(i + 120);
    sub_5E71C0((__int64)a1, a2);
  }
  v4 = v2[21];
  v5 = *(_QWORD **)v4;
  if ( *(_QWORD *)v4 )
  {
    do
    {
      a1 = (_QWORD *)v5[5];
      sub_5E71C0((__int64)a1, a2);
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
  }
  result = *(_QWORD *)(v4 + 152);
  if ( result && (*(_BYTE *)(result + 29) & 0x20) == 0 )
  {
    v7 = *(_QWORD *)(result + 144);
    if ( v7 )
    {
      do
      {
        a1 = (_QWORD *)v7;
        sub_5E7390(v7, a2);
        v7 = *(_QWORD *)(v7 + 112);
      }
      while ( v7 );
      result = *(_QWORD *)(v4 + 152);
    }
    v8 = *(_QWORD *)(result + 112);
    if ( v8 )
    {
      do
      {
        a1 = *(_QWORD **)(v8 + 120);
        *(_BYTE *)(v8 + 88) = *(_BYTE *)(v8 + 88) & 0x8F | 0x20;
        *(_BYTE *)(v8 + 136) = ((*(_BYTE *)(*(_QWORD *)v8 + 81LL) >> 1) ^ 1) & 1;
        sub_5E71C0((__int64)a1, a2);
        v8 = *(_QWORD *)(v8 + 112);
      }
      while ( v8 );
      result = *(_QWORD *)(v4 + 152);
    }
    for ( j = *(_QWORD *)(result + 104); j; j = *(_QWORD *)(j + 112) )
    {
      a1 = (_QWORD *)j;
      sub_5E71C0(j, a2);
    }
    result = *(_QWORD *)(v4 + 168);
    v11[0] = result;
    if ( result )
    {
      v10 = *(_BYTE *)(result + 8);
      if ( v10 != 3 )
        goto LABEL_19;
      a1 = v11;
      sub_72F220(v11);
      result = v11[0];
      if ( v11[0] )
      {
        v10 = *(_BYTE *)(v11[0] + 8LL);
LABEL_19:
        while ( v10 != 1 )
        {
          if ( v10 != 2 )
          {
            if ( v10 )
              sub_721090(a1);
            a1 = *(_QWORD **)(result + 32);
            goto LABEL_25;
          }
LABEL_27:
          result = *(_QWORD *)v11[0];
          v11[0] = result;
          if ( !result )
            return result;
          v10 = *(_BYTE *)(result + 8);
          if ( v10 == 3 )
          {
            a1 = v11;
            sub_72F220(v11);
            result = v11[0];
            if ( !v11[0] )
              return result;
            v10 = *(_BYTE *)(v11[0] + 8LL);
          }
        }
        a1 = *(_QWORD **)(*(_QWORD *)(result + 32) + 128LL);
LABEL_25:
        if ( a1 )
          sub_5E71C0((__int64)a1, a2);
        goto LABEL_27;
      }
    }
  }
  return result;
}
