// Function: sub_774CF0
// Address: 0x774cf0
//
__int64 __fastcall sub_774CF0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 **a4, __int64 a5)
{
  unsigned __int8 *v8; // rbx
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rax
  char v17; // al

  v8 = *a4;
  v9 = **a4;
  if ( v9 == 48 )
  {
    v14 = *((_QWORD *)v8 + 1);
    v15 = *(_BYTE *)(v14 + 8);
    if ( v15 != 1 )
    {
      if ( v15 == 2 )
      {
        *v8 = 59;
        *((_QWORD *)v8 + 1) = *(_QWORD *)(v14 + 32);
        goto LABEL_9;
      }
      if ( v15 )
        sub_721090();
      *v8 = 6;
      *((_QWORD *)v8 + 1) = *(_QWORD *)(v14 + 32);
LABEL_31:
      v12 = sub_8D2290(*((_QWORD *)v8 + 1));
      v17 = *(_BYTE *)(v12 + 140);
      if ( (unsigned __int8)(v17 - 9) <= 2u )
        goto LABEL_11;
      if ( v17 == 2 )
      {
        if ( (*(_BYTE *)(v12 + 161) & 8) == 0 )
          goto LABEL_17;
LABEL_11:
        if ( (*(_BYTE *)(v12 + 89) & 4) != 0 )
        {
          *(_BYTE *)a5 = 6;
          *(_QWORD *)(a5 + 8) = *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL);
          result = 1;
        }
        else
        {
          *(_BYTE *)a5 = 23;
          if ( (*(_BYTE *)(v12 + 89) & 2) != 0 )
            v16 = sub_72F070(v12);
          else
            v16 = *(_QWORD *)(v12 + 40);
          *(_QWORD *)(a5 + 8) = v16;
          result = 1;
        }
        goto LABEL_19;
      }
      if ( v17 != 12 )
        goto LABEL_17;
LABEL_39:
      if ( !*(_QWORD *)(v12 + 8) )
        goto LABEL_17;
      goto LABEL_11;
    }
    *v8 = 2;
    *((_QWORD *)v8 + 1) = *(_QWORD *)(v14 + 32);
LABEL_38:
    v12 = *((_QWORD *)v8 + 1);
    goto LABEL_39;
  }
  if ( v9 == 13 )
  {
    v10 = *((_QWORD *)v8 + 1);
    v11 = *(_BYTE *)(v10 + 24);
    if ( v11 == 4 )
    {
      *v8 = 8;
      v12 = *(_QWORD *)(v10 + 56);
      *((_QWORD *)v8 + 1) = v12;
      if ( (*(_BYTE *)(v12 - 8) & 1) != 0 )
        goto LABEL_8;
    }
    else
    {
      if ( v11 <= 4u )
      {
        if ( v11 != 2 )
        {
          if ( v11 == 3 )
          {
            *v8 = 7;
            v12 = *(_QWORD *)(v10 + 56);
            *((_QWORD *)v8 + 1) = v12;
            if ( (*(_BYTE *)(v12 - 8) & 1) == 0 )
              goto LABEL_10;
            goto LABEL_8;
          }
LABEL_41:
          if ( (*(_BYTE *)(v10 - 8) & 1) != 0 )
            *((_DWORD *)v8 + 4) = 0;
          goto LABEL_17;
        }
        *v8 = 2;
        v12 = *(_QWORD *)(v10 + 56);
        *((_QWORD *)v8 + 1) = v12;
        if ( (*(_BYTE *)(v12 - 8) & 1) == 0 )
          goto LABEL_39;
        *((_DWORD *)v8 + 4) = 0;
        goto LABEL_38;
      }
      if ( v11 != 20 )
        goto LABEL_41;
      *v8 = 11;
      v12 = *(_QWORD *)(v10 + 56);
      *((_QWORD *)v8 + 1) = v12;
      if ( (*(_BYTE *)(v12 - 8) & 1) != 0 )
      {
LABEL_8:
        *((_DWORD *)v8 + 4) = 0;
LABEL_9:
        v12 = *((_QWORD *)v8 + 1);
      }
    }
LABEL_10:
    if ( v12 )
      goto LABEL_11;
    goto LABEL_17;
  }
  if ( v9 <= 0xBu )
  {
    if ( v9 > 1u )
    {
      switch ( v9 )
      {
        case 2u:
          goto LABEL_38;
        case 6u:
          goto LABEL_31;
        case 7u:
        case 8u:
        case 0xBu:
          goto LABEL_9;
        default:
          goto LABEL_17;
      }
    }
    goto LABEL_17;
  }
  if ( v9 == 59 )
    goto LABEL_9;
LABEL_17:
  result = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    result = 0;
  }
LABEL_19:
  *(_DWORD *)(a5 + 16) = *((_DWORD *)v8 + 4);
  return result;
}
