// Function: sub_68A860
// Address: 0x68a860
//
void __fastcall sub_68A860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 i; // rbx
  __int64 v8; // r15
  char v9; // dl
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // rax

  v6 = a1;
  for ( i = *(_QWORD *)(a3 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v8 = *(_QWORD *)(i + 160);
  if ( (*(_BYTE *)(a3 + 207) & 0x20) != 0 )
  {
    if ( v8 == a1 || (unsigned int)sub_8D97D0(a1, *(_QWORD *)(i + 160), 0, a4, a5) )
      goto LABEL_19;
    v9 = *(_BYTE *)(a1 + 140);
    if ( v9 == 12 )
    {
      v10 = a1;
      do
      {
        v10 = *(_QWORD *)(v10 + 160);
        v9 = *(_BYTE *)(v10 + 140);
      }
      while ( v9 == 12 );
    }
    if ( v9 )
    {
      v11 = *(_BYTE *)(v8 + 140);
      if ( v11 == 12 )
      {
        v12 = v8;
        do
        {
          v12 = *(_QWORD *)(v12 + 160);
          v11 = *(_BYTE *)(v12 + 140);
        }
        while ( v11 == 12 );
      }
      if ( v11 )
      {
        if ( dword_4F04C44 != -1
          || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 6) != 0)
          || *(_BYTE *)(v13 + 4) == 12 )
        {
          if ( (unsigned int)sub_8DBE70(a1) || (unsigned int)sub_8DBE70(v8) )
          {
            if ( (unsigned int)sub_8DBE70(v8) )
            {
              if ( (unsigned int)sub_8DBE70(a1) )
                v6 = *(_QWORD *)&dword_4D03B80;
            }
            else
            {
              v6 = v8;
            }
            goto LABEL_19;
          }
        }
        a1 = 2546;
        sub_6E5ED0(2546, a2, v6, v8);
      }
    }
    v6 = sub_72C930(a1);
    goto LABEL_19;
  }
  if ( sub_624600(a1, 0, a2) )
  {
    *(_QWORD *)(i + 160) = a1;
    sub_7325D0(i, a2);
  }
  else
  {
    v6 = sub_72C930(a1);
  }
  if ( dword_4F04C58 != -1 )
    *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 760) = v8;
  *(_BYTE *)(a3 + 207) |= 0x20u;
LABEL_19:
  *(_QWORD *)(i + 160) = v6;
}
