// Function: sub_814A10
// Address: 0x814a10
//
void __fastcall sub_814A10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char v7; // al
  char v8; // al
  _BYTE *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rdi

  if ( a1 )
  {
    v6 = a1;
    do
    {
      v7 = *(_BYTE *)(v6 + 140);
      if ( (unsigned __int8)(v7 - 9) <= 2u )
      {
        if ( !(unsigned int)sub_736DD0(v6) )
        {
          v11 = *(_QWORD *)(v6 + 168);
          v12 = *(_QWORD *)(v11 + 152);
          if ( v12 && (*(_BYTE *)(v12 + 29) & 0x20) == 0 )
            sub_814600(v12, a2);
          sub_814A10(*(_QWORD *)(v11 + 216));
          goto LABEL_5;
        }
        if ( *(_BYTE *)(v6 + 140) != 2 )
          goto LABEL_5;
      }
      else if ( v7 != 2 )
      {
        goto LABEL_5;
      }
      v8 = *(_BYTE *)(v6 + 161);
      if ( (v8 & 8) != 0 )
      {
        if ( (*(_BYTE *)(v6 + 89) & 4) != 0 || (v9 = *(_BYTE **)(v6 + 40)) != 0 && v9[28] == 3 )
        {
          v9 = *(_BYTE **)(v6 + 176);
          if ( (*v9 & 1) != 0 )
          {
            v10 = *(_QWORD *)(v6 + 168);
            if ( (v8 & 0x10) != 0 )
LABEL_15:
              v10 = *(_QWORD *)(*(_QWORD *)(v6 + 168) + 96LL);
            while ( v10 )
            {
              if ( (*(_BYTE *)(v10 + 89) & 8) == 0 )
                sub_8134A0(v10, (__int64)a2, (__int64)v9, a4, a5, a6);
              v10 = *(_QWORD *)(v10 + 120);
            }
          }
        }
        else if ( (v8 & 0x10) != 0 && (**(_BYTE **)(v6 + 176) & 1) != 0 )
        {
          goto LABEL_15;
        }
      }
LABEL_5:
      v6 = *(_QWORD *)(v6 + 112);
    }
    while ( v6 );
  }
}
