// Function: sub_82F150
// Address: 0x82f150
//
__int64 __fastcall sub_82F150(_BYTE *a1)
{
  char v1; // al
  char v2; // cl
  __int64 v3; // r8
  char v4; // dl
  __int64 v5; // rax
  _BOOL4 v7; // eax
  __int64 v8; // r8

  v1 = a1[16];
  if ( v1 == 1 )
    return sub_8D70E0(*((_QWORD *)a1 + 18));
  v2 = a1[17];
  if ( v2 != 1 )
  {
    if ( v1 )
    {
      v3 = *(_QWORD *)a1;
      v4 = *(_BYTE *)(*(_QWORD *)a1 + 140LL);
      if ( v4 == 12 )
      {
        v5 = *(_QWORD *)a1;
        do
        {
          v5 = *(_QWORD *)(v5 + 160);
          v4 = *(_BYTE *)(v5 + 140);
        }
        while ( v4 == 12 );
      }
      if ( v2 == 2 && v4 )
        return v3;
    }
    return 0;
  }
  if ( v1 != 2 )
    return 0;
  v7 = sub_694910(a1);
  v8 = 0;
  if ( v7 )
    return *(_QWORD *)a1;
  return v8;
}
