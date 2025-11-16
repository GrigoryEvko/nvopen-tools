// Function: sub_7E34E0
// Address: 0x7e34e0
//
void __fastcall sub_7E34E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int16 v10; // ax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int16 v15; // dx

  if ( a2 )
  {
    v4 = *(_QWORD *)(a2 + 40);
    v5 = *(_QWORD *)(v4 + 168);
    if ( (*(_BYTE *)(v4 + 176) & 0x50) != 0 && (*(_BYTE *)(a2 + 96) & 8) == 0 && *(_QWORD *)(a2 + 128) == -1 )
    {
      *(_QWORD *)(a2 + 128) = *a3;
      v8 = *a3 + sub_7E3470(a1, a2);
      *a3 = v8;
      v9 = v8;
      v10 = *(_WORD *)(v5 + 44);
      if ( v10 != -1 )
        *a3 = v9 + (unsigned __int16)(v10 + 1);
      v5 = *(_QWORD *)(v4 + 168);
    }
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 168);
    sub_7E32B0(a1, 0, 0);
    if ( (*(_BYTE *)(a1 + 176) & 0x50) != 0 )
    {
      v14 = *a3 + ~*(_QWORD *)(v13 + 56);
      *a3 = v14;
      v15 = *(_WORD *)(v13 + 44);
      if ( v15 != -1 )
        *a3 = (unsigned __int16)(v15 + 1) + v14;
    }
    v5 = *(_QWORD *)(a1 + 168);
    v4 = a1;
  }
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6 )
  {
    do
    {
      if ( (*(_BYTE *)(v6 + 96) & 3) == 1 )
      {
        v7 = sub_8E5310(v6, a1, a2);
        sub_7E34E0(a1, v7, a3);
      }
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v6 );
    if ( !a2 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)(v4 + 168) + 16LL);
      if ( v11 )
      {
        if ( (*(_BYTE *)(v11 + 96) & 2) != 0 )
          goto LABEL_18;
        while ( 1 )
        {
          v11 = *(_QWORD *)(v11 + 16);
          if ( !v11 )
            break;
          if ( (*(_BYTE *)(v11 + 96) & 2) != 0 )
          {
LABEL_18:
            v12 = sub_8E5310(v11, a1, 0);
            sub_7E34E0(a1, v12, a3);
          }
        }
      }
    }
  }
}
