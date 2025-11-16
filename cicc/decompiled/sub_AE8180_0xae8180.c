// Function: sub_AE8180
// Address: 0xae8180
//
void __fastcall sub_AE8180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned __int8 v4; // al
  __int64 v5; // r13
  unsigned __int8 **v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // r13

  if ( a3 )
  {
    v3 = a3;
    do
    {
      v4 = *(_BYTE *)(v3 - 16);
      v5 = v3 - 16;
      if ( (v4 & 2) != 0 )
        v6 = *(unsigned __int8 ***)(v3 - 32);
      else
        v6 = (unsigned __int8 **)(v5 - 8LL * ((v4 >> 2) & 0xF));
      sub_AE8080(a1, *v6);
      v7 = *(_BYTE *)(v3 - 16);
      if ( (v7 & 2) != 0 )
      {
        if ( *(_DWORD *)(v3 - 24) != 2 )
          return;
        v8 = *(_QWORD *)(v3 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xF) != 2 )
          return;
        v8 = v5 - 8LL * ((v7 >> 2) & 0xF);
      }
      v3 = *(_QWORD *)(v8 + 8);
    }
    while ( v3 );
  }
}
